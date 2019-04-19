import math
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

class PositionalEncoder(tfk.layers.Layer):
  def __init__(self, length: int, dim: int, *args, **kwargs):
    depth_counter = np.arange(dim) // 2 * 2
    depth_matrix = np.tile(np.expand_dims(depth_counter, 0), (length, 1))
    depth_matrix = np.power(10000.0, depth_matrix / dim)
    
    phase = np.arange(dim) % 2 * math.pi / 2
    phase_matrix = np.tile(np.expand_dims(phase, 0), (length, 1))
    
    pos_counter = np.arange(length)
    pos_matrix = np.tile(np.expand_dims(pos_counter, 1), (1, dim))
    
    positional_encoder = np.sin(pos_matrix / depth_matrix + phase_matrix)
    positional_encoder = np.expand_dims(positional_encoder, 0)
    self.pe = tf.constant(positional_encoder)
    super(PositionalEncoder, self).__init__(*args, **kwargs)
  def call(self, inputs):
    fl_type = inputs.dtype
    batch_size, _, _ = tf.unstack(tf.shape(inputs))
    
    return tf.tile(tf.cast(self.pe, fl_type), [batch_size, 1, 1])
