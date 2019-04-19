# SimpleBERT
自然言語処理の勉強中で、なにやらBERTが面白そうなので、kerasの勉強も兼ねて簡易版を実装。  
トランスフォーマ部分に関しては[こちら](https://qiita.com/halhorn/items/c91497522be27bde17ce)を参考にさせていただきました。
## Requirement
使用ライブラリはmath,numpyとtensorflow  
CPU動作は現実的ではないので、GPUかTPUの実行環境
## Usage

```
from simple_bert.bert import SimpleBERT
from simple_bert.util import TextProcessor

max_length = 150
processor = TextProcessor(doc)
bert = SimpleBERT(processor.vocab_num, max_length=max_length)

x1, x2, y1, y2 = processor.build_training_data(doc, max_length=max_length)
hist = bert.pretraining(x1, x2, y1, y2)
```
docは段落・文・単語の3次リスト

## Author
Fukuzawa Taiki (taikingyo@gmail.com)