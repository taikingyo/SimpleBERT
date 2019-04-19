import tensorflow as tf
import tensorflow.keras as tfk

class MultiHeadAttention(tfk.models.Model):
  def __init__(self, dim: int, head_num: int, dropout_rate: float, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.dim = dim
    self.head_num = head_num
    self.dropout_rate = dropout_rate
    
    self.query_dense = tfk.layers.Dense(dim, use_bias=False, name='query_dense')
    self.key_dense = tfk.layers.Dense(dim, use_bias=False, name='key_dense')
    self.value_dense = tfk.layers.Dense(dim, use_bias=False, name='value_dense')
    self.out_dense = tfk.layers.Dense(dim, use_bias=False, name='out_dense')
    self.dropout = tfk.layers.Dropout(rate=dropout_rate)
  
  def call(self, inputs):
    if isinstance(inputs, list):
      if len(inputs) == 3:
        main = inputs[0]
        memory = inputs[1]
        attention_mask = inputs[2]
      elif len(inputs) == 2:
        main = inputs[0]
        memory = inputs[0]
        attention_mask = inputs[1]
      else:
        raise ValueError('unmatch list length')
      attention_mask = tf.expand_dims(attention_mask, 1)
      attention_mask = tf.expand_dims(attention_mask, 1) #[batch_size, 1, 1, m_length]
      
      query = self.query_dense(main)   #[batch_size, q_length, dim]
      key = self.key_dense(memory)     #[batch_size, m_length, dim]
      value = self.value_dense(memory) #[batch_size, m_length, dim]
    
      query = self._split(query) #[batch_size, head_num, q_length, dim // head_num]
      key = self._split(key)     #[batch_size, head_num, m_length, dim // head_num]
      value = self._split(value) #[batch_size, head_num, m_length, dim // head_num]
    
      depth = self.dim // self.head_num
      query *= depth ** -0.5
    
      logit = tf.matmul(query, key, transpose_b=True) #[batch_size, head_num, q_length, m_length]
      logit += tf.cast(attention_mask, tf.float32) * main.dtype.min
    
      weight = tf.nn.softmax(logit, name='weight')
      weight = self.dropout(weight)
    
      out = tf.matmul(weight, value) #[batch_size, head_num, q_length, dim // head_num]
      out = self._combine_head(out) #[batch_size, q_length, dim]
    
      return self.out_dense(out) #[batch_size, q_length, dim]
    else:
      raise ValueError('inputs expect list of Tensor')
    
  def _split(self, x):
    '''
    入力tensor(x)をnHeadに分割
    input_shape:  [batch_size, length, dim]
    output_shape: [batch_size, head_num, length, dim // head_num]
    '''
    
    with tf.name_scope('split'):
      batch_size, length, dim = tf.unstack(tf.shape(x))
      x = tf.reshape(x, [batch_size, length, self.head_num, self.dim // self.head_num])
      return tf.transpose(x, [0, 2, 1, 3])
    
  def _combine_head(self, x):
    '''
    入力tensor(x)の各Headを結合
    input_shape:  [batch_size, head_num, length, dim // head_num]
    output_shape: [batch_size, length, dim]
    '''
    
    with tf.name_scope('combine'):
      batch_size, _, length, _ = tf.unstack(tf.shape(x))
      x = tf.transpose(x, [0, 2, 1, 3])
      return tf.reshape(x, [batch_size, length, self.dim])
