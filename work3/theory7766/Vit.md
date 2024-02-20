#1.Transformer
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

##3.MLP Head（最终用于分类）



##4.数据集处理

1. 使用torchvision.datasets.utils中的extract_archive对从网上下载好的压缩包进行解压，得到的文件夹包含102个图像文件夹。删除
BACKGROUND_Google这个杂类。

2. 生成annotations.csv：包含图像路径以及标签名称

3. 使用sklearn中train_test_split函数划分数据集，后生成train_annotations.csv以及annotations.csv

4. 自定义数据集
特别注意，caltech101中包含灰度图像，故需先转换为三通道图像再进行transform。

##5.热力图Grad-CAM


