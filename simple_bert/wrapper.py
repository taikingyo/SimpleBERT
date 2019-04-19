import tensorflow as tf
import tensorflow.keras as tfk

class FeedForwardNetwork(tfk.models.Model):
  def __init__(self, dim: int, dropout_rate: float, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.dim = dim
    self.dropout_rate = dropout_rate
    
    self.dense_1 = tfk.layers.Dense(dim * 4, use_bias=True, activation=tf.nn.relu, name='filter')
    self.dense_2 = tfk.layers.Dense(dim, use_bias=True, name='output')
    self.dropout = tfk.layers.Dropout(rate=dropout_rate)
    
  def call(self, inputs):
    '''
    input_shape: [batch_size, length, dim]
    output_shape: [batch_size, length, dim]
    '''
    tensor = self.dense_1(inputs)
    tensor = self.dropout(tensor)
    return self.dense_2(tensor)

class LayerNormalization(tfk.layers.Layer):
  def build(self, input_shape):
    dim = input_shape[-1]
    self.scale = self.add_weight('layer_norm_scale', shape=[dim], initializer=tf.ones_initializer())
    self.bias = self.add_weight('layer_norm_bial', [dim], initializer=tf.zeros_initializer())
    super().build(input_shape)
    
  def call(self, x: tf.Tensor, epsilon: float = 1e-6):
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) / tf.sqrt(variance + epsilon)
    
    return norm_x * self.scale + self.bias
  
class ResidualNormalizationWrapper(tfk.models.Model):
  def __init__(self, layer: tfk.layers.Layer, dropout_rate: float, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.layer = layer
    self.norm = LayerNormalization()
    self.dropout =  tfk.layers.Dropout(rate=dropout_rate)
    
  def call(self, input):
    if isinstance(input, list):
      head = input[0]
      tail = input[1:]
      head = self.norm(head)
      tensor = self.layer([head] + tail)
      tensor = self.dropout(tensor)
      tensor += head
    else:
      tensor = self.norm(input)
      tensor = self.layer(tensor)
      tensor = self.dropout(tensor)
      tensor += input
    return tensor
