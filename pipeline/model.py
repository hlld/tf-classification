import tensorflow as tf
from pipeline.utils import Conv, Bottleneck, GlobalPool2d


class ResNet(tf.keras.Model):
    def __init__(self,
                 num_classes=1000,
                 model_type='resnet18',
                 l2_alpha=0,
                 name='ResNet',
                 **kwargs):
        super(ResNet, self).__init__(name=name, **kwargs)
        if model_type == 'resnet18':
            block_num = [2, 2, 2, 2]
        elif model_type == 'resnet34':
            block_num = [3, 4, 6, 3]
        elif model_type == 'resnet50':
            block_num = [3, 4, 6, 3]
        elif model_type == 'resnet101':
            block_num = [3, 4, 23, 3]
        elif model_type == 'resnet152':
            block_num = [3, 8, 36, 3]
        else:
            raise ValueError('Unknown type %s' % model_type)
        if model_type in ['resnet18', 'resnet34']:
            filter_num = [64, 128, 256, 512]
            shortcut = [False, True, True, True]
            use_residual = True
        else:
            filter_num = [256, 512, 1024, 2048]
            shortcut = [True, True, True, True]
            use_residual = False
        down_sample = [False, True, True, True]
        self.stem = tf.keras.models.Sequential([
            Conv(64,
                 kernel_size=7,
                 strides=2,
                 activation='relu',
                 l2_alpha=l2_alpha),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2,
                                      padding='same')
        ])
        layers = []
        for num, filters, conv, down in zip(block_num,
                                            filter_num,
                                            shortcut,
                                            down_sample):
            layers.append(self._make_layer(filters,
                                           shortcut_conv=conv,
                                           down_sample=down,
                                           use_residual=use_residual,
                                           l2_alpha=l2_alpha,
                                           num_blocks=num))
        self.blocks = tf.keras.models.Sequential(layers)
        self.logits = tf.keras.models.Sequential([
            GlobalPool2d('avg', keep_dims=False),
            tf.keras.layers.Dense(num_classes)
        ])
        self.max_stride = 32
        self.num_classes = num_classes
        self.model_type = model_type

    @staticmethod
    def _make_layer(filters,
                    shortcut_conv,
                    down_sample,
                    use_residual,
                    l2_alpha,
                    num_blocks):
        layers = [Bottleneck(filters,
                             shortcut_conv=shortcut_conv,
                             down_sample=down_sample,
                             use_residual=use_residual,
                             activation='relu',
                             l2_alpha=l2_alpha)]
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(filters,
                                     shortcut_conv=False,
                                     down_sample=False,
                                     use_residual=use_residual,
                                     activation='relu',
                                     l2_alpha=l2_alpha))
        return tf.keras.models.Sequential(layers)

    def call(self,
             inputs,
             training=None,
             mask=None):
        inputs = self.stem(inputs, training=training)
        inputs = self.blocks(inputs, training=training)
        return self.logits(inputs)
