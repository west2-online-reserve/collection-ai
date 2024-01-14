# IMDB SENTIMENT ANALYSIS项目说明

#### 项目说明

- 在测试集上可以达到**99.45%**的正确率(训练集只有98.71%比训练集还高，看到结果时都吓到我了)
- 后面我把代码上传到Colab上调了调参跑，最后可以达到99.89%的正确率，但是我没有再分出一个校验集，不知道会不会存在过拟合测试集的可能，毕竟我是当测试集准确率最高时保存模型, [Colab链接](https://colab.research.google.com/drive/1R3V9ySsDlGwDyg9sql50MIMxfEb024za?usp=sharing)
- 使用的模型是双向LSTM模型，确实有点强

#### 模型准确率说明

![image-20240113211657919](D:/project/wjord/notes/%E5%9B%BE%E7%89%87%E5%AD%98%E5%82%A8/image-20240113211657919.png)

![image-20240113211639853](D:/project/wjord/notes/%E5%9B%BE%E7%89%87%E5%AD%98%E5%82%A8/image-20240113211639853.png)

colab上的

![image-20240114124431612](./../../../wjord/notes/%E5%9B%BE%E7%89%87%E5%AD%98%E5%82%A8/image-20240114124431612.png)

#### 完成内容

- [x] 自主完成IMDB数据集的正负语义判断

- [x] 使用matplotlib绘制训练集的精确度(train_acc)和损失(train_loss)以及测试集的精确度(train_acc)和损失(train_loss)的曲线

- [x] 完成了数据集的处理，并实例化为了dataset类
- [x] 没有使用预训练模型