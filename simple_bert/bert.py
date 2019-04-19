import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras import backend as K
from simple_bert.encoder import Encoder
from simple_bert.embedding import PositionalEncoder

class SimpleBERT:
  def __init__(
      self,
      vocab_num: int,
      max_length: int,
      unit_num: int=6,
      hopping_num: int=6,
      head_num: int=8,
      dim: int=256,
      dropout_rate: float=0.1,
      init_lr: float=1e-4,
  ) -> None:
    self.vocab_num = vocab_num
    self.max_length = max_length
    self.encoder_list = []
    for i in range(unit_num):
      self.encoder_list.append(Encoder(self.vocab_num, hopping_num, head_num, dim, dropout_rate, max_length, name=f'encoder_{i}'))
      
    inputs1 = tfk.Input(shape=(max_length,), dtype='int32', name='main_input') #[batch_size, max_length]
    inputs2 = tfk.Input(shape=(max_length,), dtype='int32', name='segment_type') #[batch_size, max_length]
    mask = tfk.layers.Lambda(lambda x: K.cast(K.not_equal(x, 0), tf.float32))(inputs1)
    mask = tfk.layers.Lambda(lambda x: K.tile(K.expand_dims(x, -1), [1, 1, dim]))(mask)
    pad = tfk.layers.Lambda(lambda x: K.cast(K.equal(x, 0), tf.float32))(inputs1)
    
    token_embedded = tfk.layers.Embedding(input_dim=self.vocab_num, output_dim=dim, input_length=max_length)(inputs1)
    token_embedded = tfk.layers.Lambda(lambda x: x * (dim ** 0.5))(token_embedded)
    position_embedded = PositionalEncoder(max_length, dim)(token_embedded)
    segment_embedded = tfk.layers.Embedding(input_dim=3, output_dim=dim, input_length=max_length)(inputs2)
    embedded = tfk.layers.Add()([token_embedded, position_embedded, segment_embedded])
    embedded = tfk.layers.Dropout(rate=dropout_rate)(embedded)
    embedded = tfk.layers.Multiply()([embedded, mask])
    
    query = embedded
    for encoder in self.encoder_list:
      query = encoder([query, pad])
    
    flat = tfk.layers.Flatten()(query)
    sentence_shuffle_y_ = tfk.layers.Dense(1, activation='sigmoid', name='shuffle')(flat)
    random_mask_y_ = tfk.layers.Dense(self.vocab_num, activation='softmax', name='mask')(flat)
    
    self.bert = tfk.models.Model(inputs=[inputs1, inputs2], outputs=[sentence_shuffle_y_, random_mask_y_])
    self.bert.compile(
        optimizer=tfk.optimizers.Adam(lr=init_lr, decay=0.01),
        loss={'shuffle': 'binary_crossentropy', 'mask': 'categorical_crossentropy'},
        loss_weights={'shuffle': 0.8, 'mask': 0.2},
        metrics=['accuracy'])
    
  def pretraining(self, x1, x2, y1, y2, batch_size=32, epochs=1):
    hist = self.bert.fit([x1, x2], [y1, y2], batch_size=batch_size, epochs=epochs)
    return hist
    
  def save_weights(self, file):
    self.bert.save_weights(file)
    
  def load_weights(self, file):
    self.bert.load_weights(file)
