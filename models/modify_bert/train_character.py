import tensorflow as tf
from activate_function import gelu_activate_fn

class MLMPredictionLayer(tf.keras.layers.Layer):
    def __init__(
        self, 
        hidden_size,
        vocab_size,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.2),
        activation_fn=gelu_activate_fn,
        layer_norm_eps=1e-12,
        **kwargs):
        
        self.vocab_size = vocab_size
        self.Dense = tf.keras.layers.Dense(
            hidden_size,
            kernel_initializer=kernel_initializer,
            name='dense',
        )
        self.Activation = tf.keras.layers.Activation(activation_fn)
        self.LayerNorm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps,
            name='layer_norm',
        )

    def call(self, tensor_in):
        _tensor = self.Dense(tensor_in)
        _tensor = self.Activation(_tensor)
        _tensor = self.LayerNorm(_tensor)
        return _tensor
