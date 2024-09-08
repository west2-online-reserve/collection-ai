一开始尝试使用ViT_Base模型训练，发现效果并不理想，最后准确率只有0.5左右。之后在原基础上尝试了串行block和并行block组合的结构，准确率变化不大。用grad-cam发现模型几乎找不到特征。

ViT模型没有CNN模型inductive bias的特点，在小数据集训练时效果不佳。小数据集训练下的ViT模型不能很好的捕捉到图像的局部信息。

之后又尝试了T2T_ViT模型，在同样条件下训练相同epoch后正确率到达0.6。用grad-cam发现模型可以认识大致特征。

![image-20240301174916835](C:\Users\blue\AppData\Roaming\Typora\typora-user-images\image-20240301174916835.png)

