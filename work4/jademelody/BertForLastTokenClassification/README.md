# BertForLastTokenClassification

微调huggingface上的bert-base-chinese模型，实现中文文本生成

使用预训练的BERT模型，通过对其返回的最后一个token进行分类，实现文本生成

### 数据集准备 DataPrepare

Notice: 你的数据集应该为一个json文件，标签1为'sentence_1'作为输入语句，标签2为'sentence_2'作为要生成的语句，需要将每行作为一个独立的json对象来存储（也就是lines=True的保存方式）

示例：

```json
{"sentence_1": "今天天气晴朗。", "sentence_2": "阳光明媚，心情愉快。"}
{"sentence_1": "我喜欢读书。", "sentence_2": "知识改变命运。"}
{"sentence_1": "这部电影很精彩。", "sentence_2": "值得一看。"}
```

