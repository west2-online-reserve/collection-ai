#1.Vision Transformer
先将图片处理为patches输入Embedding层,获得token。

##1.Linear Projection of Flatten Patches(Embedding层）
对于标准的Transformer模块，要求输入的是token(向量)序列，即二维矩阵[num_token,token_dim]

在代码实现中，直接通过一个卷积层来实现以ViT-B/16为例，使用卷积核大小为16x16，stride为16，卷积核个数为768
[224,224,3] ->[14,14, 768] ->[196,768]

在输入Transformer Encoder之前需要加上[class]token以及Position Embedding，都是可训练参数

拼接[class]token: Cat([1, 768],[196,768])->[197,768]

叠加Position Embedding:[197,768] ->[197,768]


在Transformer Encode层之前有个dropout层，后有个layer Norm

##2.Transformer Encode
(Layer Norm->Muti-Head Attetion->Dropout/DropPath）残差块->(Layer Norm->MLP->Dropout/DropPath)残差块
其中，MLP Block:

	# ImageNet21K
	197×768->Linear(197×3072)->GELU->Dropuout->Linear(197×768)->Dropout
	# ImageNet1K
	Linear(197×768)->Dropout

Muti-Head Attetion:
Attention(Q,K,V) = softmax(Q*K^T/dk^0.5)V

计算完softmax(Q*K^T/dk^0.5)先经过一个dropout层，再与V相乘，然后由于数据集较小，直接过一个简单的Linear层，再过一个dropout层。


##3.MLP Head（最终用于分类）
一个简单的Linear层，靠交叉熵函数实现分类

##4.数据集处理

1. 使用torchvision.datasets.utils中的extract_archive对从网上下载好的压缩包进行解压，得到的文件夹包含102个图像文件夹。删除
BACKGROUND_Google这个杂类。

2. 生成annotations.csv：包含图像路径以及标签名称

3. 使用sklearn中train_test_split函数划分数据集，后生成train_annotations.csv以及annotations.csv

4. 自定义数据集
特别注意，caltech101中包含灰度图像，故需先转换为三通道图像再进行transform。

##5.模型参数
	Model  Patch_Size Layers Hidden_Size MLP_size Headers 
	ViT-Base 16*16 12 768 3072 12
	ViT-Large 16*16 24 1024 4096 16
##6.热力图Grad-CAM
1. 选取最后一个Encoder Block的第一个layernorm作为目标层。
2. python默认图片为BGR，需转换为RGB格式
3. 图片进行居中处理到宽高均为224
4. 由于vit模型会将图片切成patches，并且在前面加上class token，需要将model去掉class token，并且恢复图片



