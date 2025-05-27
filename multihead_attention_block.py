import tensorflow as tf
from tensorflow.keras import layers

class MHSABlock(layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(MHSABlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mhsa = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.norm1 = layers.LayerNormalization()
        self.ffn_dense1 = layers.Dense(4 * key_dim, activation='relu')
        self.ffn_dense2 = None  # initialized in build
        self.norm2 = layers.LayerNormalization()

    def build(self, input_shape):
        self.ffn_dense2 = layers.Dense(input_shape[-1])

    def call(self, x):
        attn_output = self.mhsa(x, x)
        x = self.norm1(x + attn_output)
        ffn = self.ffn_dense1(x)
        ffn = self.ffn_dense2(ffn)
        x = self.norm2(x + ffn)
        return x

    def get_config(self):
        config = super(MHSABlock, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
