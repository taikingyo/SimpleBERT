import numpy as np

class TextProcessor:
  def __init__(self, doc, unk_lim: int=1):
    self.unk_lim = unk_lim
    self._build_dictionary(doc)
    
  def _build_dictionary(self, doc):
    #単語文字列から整数値への変換辞書作成
    
    words = [] #全単語リスト
    for context in doc:
      for tolk in context:
        words += tolk
    vocab = sorted(list(set(words))) #語彙リスト
    
    #低頻度後の置き換え処理
    word2indice = dict((w, i) for i, w in enumerate(vocab))
    cnt = np.zeros(len(vocab))
    for i in range(len(words)):
      cnt[word2indice[words[i]]] += 1
    vocab_unk = [] #低頻度語・未知語リスト（未使用）
    for i in range(len(cnt)):
      if cnt[i] <= self.unk_lim:
        vocab_unk.append(vocab[i])
        vocab[i] = '<UNK>'
        
    #制御コード＋語彙リスト
    vocab = ['<PAD>', '<MSK>', '<CLS>', '<SEP>'] + sorted(list(set(vocab)))
    if not '<UNK>' in vocab:
      vocab += ['<UNK>']
      
    #語彙辞書
    word2indice = dict((w, i) for i, w in enumerate(vocab))
    indice2word = dict((i, w) for i, w in enumerate(vocab))
    
    self.vocab = vocab
    self.vocab_unk = vocab_unk
    self.vocab_num = len(vocab)
    self.word2indice = word2indice
    self.indice2word = indice2word
    
  def build_training_data(self, data, max_length):
    #sentence shuffle
    double_sentences = [] #ベースの隣接文
    single_sentences = [] #シャッフル用予備文
    for context in data:
      if len(context) == 1:
        single_sentences.append(context[0])
      else:
        double_sentences += tuple(zip(context[:-1], context[1:]))
        
    if len(single_sentences) < 1:
      np.random.shuffle(double_sentences)
      (s1, s2) = double_sentences[0]
      single_sentences.append(s1)
      single_sentences.append(s2)
      double_sentences = double_sentences[1:]
    np.random.shuffle(single_sentences)
    
    #確率0.25でそれぞれ隣接文の前半・後半を予備文に置き換え（そのままでは使えない単文を優先）
    sentences = []
    y1 = np.zeros(len(double_sentences), dtype=int) #隣接文判別ターゲット
    for i, (s1, s2) in enumerate(double_sentences):
      r = np.random.rand()
      if r < 0.25:
        single_sentences.append(s1)
        s1 = single_sentences[0]
        single_sentences = single_sentences[1:]
      elif r < 0.5:
        single_sentences.append(s2)
        s2 = single_sentences[0]
        single_sentences = single_sentences[1:]
      else:
        y1[i] = 1
      sentence = ['<CLS>'] + s1 + ['<SEP>'] + s2 + ['<SEP>']
      if len(sentence) > max_length:
        sentence = sentence[:max_length]
      sentences.append(sentence)
    y1 = np.expand_dims(y1, axis=-1)
      
    #random masking
    x2 = np.zeros((len(sentences), max_length), dtype=int) #セグメントタイプ(文１:1、文２:2、パディング:0)
    y2 = np.zeros((len(sentences), self.vocab_num), dtype=int) #ランダムマスクターゲット(n-hot vector)
    masked_sentences = []
    for i, sentence in enumerate(sentences):
      label = 1
      masked = []
      for j, word in enumerate(sentence):
        x2[i][j] = label
        if not word in self.word2indice:
          word = '<UNK>'
        elif word == '<SEP>':
          label = 2
        elif not(word.startswith('<') and word.endswith('>')):
          if np.random.rand() < 0.13:
            y2[i][self.word2indice[word]] = 1
            word = '<MSK>'
        masked.append(word)
      masked_sentences.append(masked)
        
    x1 = self.toIndices(masked_sentences, max_length) #main input
      
    return x1, x2, y1, y2
    
  def toIndices(self, sentences, max_length):
    #文字列からone-hot表現へ変換
    indices = np.zeros((len(sentences), max_length), dtype=int)
    for i, sentence in enumerate(sentences):
      if len(sentence) > max_length:
        sentence = sentence[:max_length]
      for j, word in enumerate(sentence):
        if not word in self.word2indice:
          word = '<UNK>'
        indices[i][j] = self.word2indice[word]
          
    return indices
    
  def toWords(self, indices):
    #one-hot表現から文字列へ変換
    sentences = []
    for line in indices:
      sentence = []
      for indice in line:
        if indice == 0:
          break
        sentence.append(self.indice2word[indice])
      sentences.append(sentence)
        
    return sentences
