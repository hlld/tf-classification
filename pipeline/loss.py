import tensorflow as tf


class CategoricalCrossentropy(tf.losses.Loss):
    def __init__(self):
        super(CategoricalCrossentropy, self).__init__(
            reduction='none',
            name='CategoricalCrossentropy')

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_pred)
        return cross_entropy


class BinaryCrossentropy(tf.losses.Loss):
    def __init__(self):
        super(BinaryCrossentropy, self).__init__(
            reduction='none',
            name='BinaryCrossentropy')

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred)
        return cross_entropy
