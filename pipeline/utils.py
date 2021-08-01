import tensorflow as tf


class GlobalPool2d(tf.keras.layers.Layer):
    def __init__(self,
                 pooling_type='avg',
                 axis=(1, 2),
                 keep_dims=True,
                 name=None,
                 **kwargs):
        super(GlobalPool2d, self).__init__(name=name, **kwargs)
        assert pooling_type in ['avg', 'max']
        self.pooling_type = pooling_type
        self.axis = axis
        self.keep_dims = keep_dims

    def call(self,
             inputs,
             training=None):
        if self.pooling_type == 'avg':
            return tf.reduce_mean(inputs,
                                  axis=self.axis,
                                  keepdims=self.keep_dims)
        else:
            return tf.reduce_max(inputs,
                                 axis=self.axis,
                                 keepdims=self.keep_dims)


class Activation(tf.keras.layers.Layer):
    def __init__(self,
                 activation_type='relu',
                 name=None,
                 **kwargs):
        super(Activation, self).__init__(name=name, **kwargs)
        assert activation_type in ['relu', 'relu6', 'none']
        self.activation_type = activation_type

    def call(self,
             inputs,
             training=None):
        if self.activation_type == 'relu':
            return tf.nn.relu(inputs)
        elif self.activation_type == 'relu6':
            return tf.nn.relu6(inputs)
        return inputs


class Conv(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='same',
                 activation='relu',
                 kernel_init=tf.keras.initializers.GlorotUniform(),
                 bias_init=tf.keras.initializers.Constant(0),
                 l2_alpha=0,
                 name=None,
                 **kwargs):
        super(Conv, self).__init__(name=name, **kwargs)
        regularizer = None
        if l2_alpha > 0:
            regularizer = tf.keras.regularizers.l2(l2=l2_alpha)
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding,
                                           use_bias=False,
                                           kernel_initializer=kernel_init,
                                           bias_initializer=bias_init,
                                           kernel_regularizer=regularizer,
                                           name=name)
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = Activation(activation)

    def call(self,
             inputs,
             training=None):
        inputs = self.conv(inputs)
        inputs = self.bn(inputs, training=training)
        return self.act(inputs)


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 shortcut_conv=False,
                 down_sample=False,
                 use_residual=False,
                 activation='relu',
                 l2_alpha=0,
                 name=None,
                 **kwargs):
        super(Bottleneck, self).__init__(name=name, **kwargs)
        stride = 2 if down_sample else 1
        self.shortcut_conv = None
        if shortcut_conv:
            self.shortcut_conv = Conv(filters,
                                      kernel_size=1,
                                      strides=stride,
                                      activation='none',
                                      l2_alpha=l2_alpha)
        if use_residual:
            layers = [Conv(filters,
                           kernel_size=3,
                           strides=stride,
                           activation=activation,
                           l2_alpha=l2_alpha),
                      Conv(filters,
                           kernel_size=3,
                           strides=1,
                           activation='none',
                           l2_alpha=l2_alpha)]
        else:
            mid_filters = filters // 4
            layers = [Conv(mid_filters,
                           kernel_size=1,
                           strides=1,
                           activation=activation,
                           l2_alpha=l2_alpha),
                      Conv(mid_filters,
                           kernel_size=3,
                           strides=stride,
                           activation=activation,
                           l2_alpha=l2_alpha),
                      Conv(filters,
                           kernel_size=1,
                           strides=1,
                           activation='none',
                           l2_alpha=l2_alpha)]
        self.sequential = tf.keras.models.Sequential(layers)
        self.act = Activation(activation)

    def call(self,
             inputs,
             training=None):
        shortcut = inputs
        inputs = self.sequential(inputs, training=training)
        if self.shortcut is not None:
            shortcut = self.shortcut_conv(shortcut, training=training)
        inputs = inputs + shortcut
        return self.act(inputs)
