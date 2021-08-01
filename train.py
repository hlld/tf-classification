from absl import app, flags
from absl.flags import FLAGS
import os
import tensorflow as tf
import tensorflow.keras.mixed_precision as mp
from tqdm import tqdm
from pipeline.model import ResNet
from pipeline.loss import CategoricalCrossentropy
from pipeline.dataset import Imagefolder


def train_network(_argv):
    print('Training %s for %d epochs' % (FLAGS.model_type,
                                         FLAGS.epochs))
    # Configure dynamic memory growth
    tf.config.set_soft_device_placement(True)
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, enable=True)

    devices = FLAGS.device.strip().split(',')
    devices = ['/gpu:%d' % int(device_id) for device_id in devices]
    if len(devices) == 1:
        strategy = tf.distribute.OneDeviceStrategy(devices[0])
    else:
        strategy = tf.distribute.MirroredStrategy(devices)
    print('Number of devices %d' % strategy.num_replicas_in_sync)

    # Linear scale learning rate
    if strategy.num_replicas_in_sync > 1:
        FLAGS.batch_size *= strategy.num_replicas_in_sync
        FLAGS.initial_lr *= strategy.num_replicas_in_sync
        print('Total batch size %d' % FLAGS.batch_size)
        print('Initial learning rate %.4f' % FLAGS.initial_lr)

    trainset = Imagefolder(FLAGS.data_root,
                           data_split='train',
                           input_size=FLAGS.input_size,
                           data_augment=FLAGS.data_augment)
    train_samples = len(trainset.samples)
    train_loader = trainset.make_dataset(FLAGS.batch_size,
                                         shuffle=True,
                                         iter_count=None,
                                         drop_remainder=False,
                                         buffer_size=0)
    train_loader = strategy.experimental_distribute_dataset(train_loader)
    train_batch = (train_samples + FLAGS.batch_size - 1) // FLAGS.batch_size
    testset = Imagefolder(FLAGS.data_root,
                          data_split='val',
                          input_size=FLAGS.input_size,
                          data_augment=False)
    test_samples = len(testset.samples)
    test_loader = testset.make_dataset(FLAGS.batch_size,
                                       shuffle=False,
                                       iter_count=test_samples,
                                       drop_remainder=False,
                                       buffer_size=0)
    test_loader = strategy.experimental_distribute_dataset(test_loader)

    with strategy.scope():
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        accuracy = tf.keras.metrics.Accuracy(name='accuracy')
        criterion = CategoricalCrossentropy()
        model = ResNet(FLAGS.num_classes,
                       model_type=FLAGS.model_type,
                       l2_alpha=FLAGS.weight_decay)
        if len(FLAGS.lr_steps) == 0:
            FLAGS.lr_steps = [0.33, 0.66]
        total_batch = FLAGS.epochs * train_batch
        boundaries = [round(total_batch * FLAGS.lr_steps[0]),
                      round(total_batch * FLAGS.lr_steps[1])]
        values = [FLAGS.initial_lr,
                  FLAGS.initial_lr * 0.1,
                  FLAGS.initial_lr * 0.01]
        lr_schedule = \
            tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries,
                                                                 values=values)
        optimizer = tf.keras.optimizers.SGD(lr_schedule,
                                            momentum=FLAGS.momentum)
        optimizer = mp.experimental.LossScaleOptimizer(optimizer,
                                                       loss_scale='dynamic')
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        if FLAGS.weights:
            status = checkpoint.restore(FLAGS.weights)
            status.expect_partial()
            print('Restored model from %s' % FLAGS.weights)

    def _train_single_step(inputs):
        images, labels = inputs
        with tf.GradientTape() as tape:
            outputs = model(images, training=True)
            labels = tf.one_hot(labels, FLAGS.num_classes)
            per_sample_loss = criterion(labels, outputs)
            loss = tf.nn.compute_average_loss(
                per_sample_loss,
                global_batch_size=FLAGS.batch_size)
            # Add scaled regularization losses
            l2_loss = tf.nn.scale_regularization_loss(sum(model.losses))
            loss += l2_loss
            # Apply mixed precision loss scaling
            scaled_loss = optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(scaled_loss,
                                  model.trainable_variables)
        # Apply mixed precision gradients scaling
        gradients = optimizer.get_unscaled_gradients(gradients)
        optimizer.apply_gradients(zip(gradients,
                                      model.trainable_variables))
        train_loss.update_state(loss)
        return loss

    @tf.function(experimental_relax_shapes=True)
    def _distributed_train_step(inputs):
        per_replica_losses = strategy.run(_train_single_step,
                                          args=(inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM,
                               per_replica_losses,
                               axis=None)

    def _eval_single_step(inputs):
        images, labels = inputs
        outputs = model(images, training=False)
        labels = tf.one_hot(labels, FLAGS.num_classes)
        per_sample_loss = criterion(labels, outputs)
        loss = tf.nn.compute_average_loss(
            per_sample_loss,
            global_batch_size=FLAGS.batch_size)
        val_loss.update_state(loss)
        return outputs

    @tf.function(experimental_relax_shapes=True)
    def _distributed_eval_step(inputs):
        per_replica_outputs = strategy.run(_eval_single_step,
                                           args=(inputs,))
        outputs = strategy.experimental_local_results(per_replica_outputs)
        labels = strategy.experimental_local_results(inputs[1])
        outputs = tf.concat(outputs, axis=0)
        outputs = tf.argmax(outputs, axis=-1)
        labels = tf.concat(labels, axis=0)
        labels = tf.argmax(labels, axis=-1)
        accuracy.update_state(labels, outputs)

    def _write_checkpoint():
        ckpt_prefix = os.path.join(FLAGS.save_path,
                                   '%s_%d_%.4f_%.4f_%.4f.ckpt' % (
                                       FLAGS.model_type,
                                       epoch,
                                       train_loss.result(),
                                       val_loss.result(),
                                       accuracy.result()))
        checkpoint.save(ckpt_prefix)
        train_loss.reset_states()
        val_loss.reset_states()
        accuracy.result()

    for epoch in range(FLAGS.epochs):
        print(('\n' + '%10s' * 3) % ('Epoch',
                                     'loss',
                                     'step'))
        progress = tqdm(range(train_batch), total=train_batch)
        train_sampler = iter(train_loader)
        for _ in progress:
            inputs = next(train_sampler)
            _distributed_train_step(inputs)
            desc = ('%10s' + '%10.4g' * 2) % (
                '%g/%g' % (epoch, FLAGS.epochs - 1),
                train_loss.result(),
                optimizer.lr(optimizer.iterations))
            progress.set_description(desc)

        desc = ('%10s' * 2) % ('Top1', 'loss')
        for inputs in tqdm(test_loader, desc=desc):
            _distributed_eval_step(inputs)
        print(('%10.4g' * 2) % (accuracy.result(),
                                val_loss.result()))
        _write_checkpoint()


if __name__ == '__main__':
    flags.DEFINE_string(name='model_type', default='resnet18',
                        help='model type')
    flags.DEFINE_string(name='weights', default='',
                        help='pre-trained weights')
    flags.DEFINE_string(name='data_root', default='',
                        help='dataset root')
    flags.DEFINE_integer(name='input_size', default=224,
                         help='input size')
    flags.DEFINE_integer(name='num_classes', default=1000,
                         help='number of classes')
    flags.DEFINE_boolean(name='data_augment', default=True,
                         help='data augmentation')
    flags.DEFINE_integer(name='epochs', default=90,
                         help='training epochs')
    flags.DEFINE_integer(name='batch_size', default=256,
                         help='total batch size')
    flags.DEFINE_string(name='save_path', default='./results',
                        help='save path')
    flags.DEFINE_list(name='lr_steps', default=[],
                      help='learning rate decay steps')
    flags.DEFINE_float(name='initial_lr', default=0.1,
                       help='initial learning rate')
    flags.DEFINE_float(name='momentum', default=0.9,
                       help='SGD momentum')
    flags.DEFINE_float(name='weight_decay', default=0.0001,
                       help='weight decay')
    flags.DEFINE_string(name='device', default='0',
                        help='CUDA devices')
    app.run(train_network)
