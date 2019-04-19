import tensorflow as tf
import tensorflow.keras as tfk
from simple_bert.attention import MultiHeadAttention
from simple_bert.wrapper import ResidualNormalizationWrapper
from simple_bert.wrapper import LayerNormalization
from simple_bert.wrapper import FeedForwardNetwork

class Encoder(tfk.models.Model):
  def __init__(
    self,
    vocab: int,
    hopping_num: int,
    head_num: int,
    dim: int,
    dropout_rate: float,
    max_length: int,
    *args,
    **kwargs
  ):
    super(Encoder, self).__init__(*args, **kwargs)
    self.hopping_num = hopping_num
    self.head_num = head_num
    self.dim = dim
    self.dropout_rate = dropout_rate
    
    self.attention_list: List[List[tfk.models.Model]] = []
    
    for _ in range(hopping_num):
      attention = MultiHeadAttention(dim, head_num, dropout_rate, name='attention')
      ffn = FeedForwardNetwork(dim, dropout_rate, name='ffn')
      self.attention_list.append([
          ResidualNormalizationWrapper(attention, dropout_rate, name='attention_wrapper'),
          ResidualNormalizationWrapper(ffn, dropout_rate, name='ffn_wrapper')
      ])
      
    self.normalization = LayerNormalization()
    
  def call(self, inputs):
    query = inputs[0]
    mask = inputs[1]
    
    for i, layer in enumerate(self.attention_list):
      attention, ffn = tuple(layer)
      with tf.name_scope(f'hopping_{i}'):
        query = attention([query, mask])
        query = ffn(query)
        
    return self.normalization(query)
