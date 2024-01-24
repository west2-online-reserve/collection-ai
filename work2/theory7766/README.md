数据集：CIFAR100
输入：32×32×3的图像
评价标准：准确率
###01.py文件
是基于ResNet残差神经网络实现。

它在ResNet18基础上进行小小调整。

网络结构为

[Conv2d.64@3×3/stride=1,->BatchNorm2d(64)->ReLU]->
[MaxPool2d 1×1/stride=1]->
[Residual Block ×2]->
[Residual Block ×2]->
[Residual Block ×2]->
[Residual Block ×2]->
[AvgPool2d 1×1]->
[Linear(512×100)]->
[Dropout(0.3)]

而每个残差块ResBlock由2个卷积层构成，具体结构为
[Conv2d(k=3×3)->BatchNorm2d->ReLU]->
[Conv2d(k=3×3)->BatchNorm2d]

	ResNet(
	  (conv1): Sequential(
	    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	    (2): ReLU()
	    (3): MaxPool2d(kernel_size=1, stride=1, padding=0, dilation=1, ceil_mode=False)
	  )
	  (layer1): Sequential(
	    (0): ResBlock(
	      (block_conv): Sequential(
	        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	        (2): ReLU()
	        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	        (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	      )
	      (shortcut): Sequential()
	    )
	    (1): ResBlock(
	      (block_conv): Sequential(
	        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	        (2): ReLU()
	        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	        (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	      )
	      (shortcut): Sequential()
	    )
	  )
	  (layer2): Sequential(
	    (0): ResBlock(
	      (block_conv): Sequential(
	        (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
	        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	        (2): ReLU()
	        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	        (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	      )
	      (shortcut): Sequential(
	        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
	        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	      )
	    )
	    (1): ResBlock(
	      (block_conv): Sequential(
	        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	        (2): ReLU()
	        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	        (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	      )
	      (shortcut): Sequential()
	    )
	  )
	  (layer3): Sequential(
	    (0): ResBlock(
	      (block_conv): Sequential(
	        (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
	        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	        (2): ReLU()
	        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	        (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	      )
	      (shortcut): Sequential(
	        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
	        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	      )
	    )
	    (1): ResBlock(
	      (block_conv): Sequential(
	        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	        (2): ReLU()
	        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	        (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	      )
	      (shortcut): Sequential()
	    )
	  )
	  (layer4): Sequential(
	    (0): ResBlock(
	      (block_conv): Sequential(
	        (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
	        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	        (2): ReLU()
	        (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	        (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	      )
	      (shortcut): Sequential(
	        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
	        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	      )
	    )
	    (1): ResBlock(
	      (block_conv): Sequential(
	        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	        (2): ReLU()
	        (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	        (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	      )
	      (shortcut): Sequential()
	    )
	  )
	  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
	  (linear): Linear(in_features=512, out_features=100, bias=True)
	  (dropout): Dropout(p=0.3, inplace=False)
	)

损失函数loss：交叉熵函数CrossEntropyLoss()。


优化器：SGD/Adam

最终测试准确率：15轮 52.49%

03.py是对resnet18模型进行微调


CIFAR100_resnet1.pth是01.py的新参数，为训练15轮模型，准确率达到

02.py文件是基于GRU实现。

四层卷积层：
在4层卷积层网络上，使用SGD lr=0.005 20轮 36.34%

在4层卷积层网络上+1，使用Adam lr=0.001 5轮 40.67%


'CIFAR100_gru__3.pth 在4层卷积层网络上+1，使用Adam lr=0.00001 10轮 63.46%



'CIFAR100_gru__1.pth'在4层卷积层网络上+1，使用Adam lr=0.001 5轮  40.67%
'CIFAR100_gru__2.pth 在1基础上，使用Adam lr=0.0005 5轮 48.23%

3：Adam lr=0.0001 10轮（2）
49.82

4：Adam lr=0.0001 10轮（3）
50.4

5：Adam lr=0.00005 10轮（4）



'CIFAR100_resnet1.pth'# 使用0.01 SGD优化15轮，52.49%

'CIFAR100_resnet4.pth'在1的基础上，使用0.0001 Adam优化4*4轮，74.8%

'CIFAR100_resnet6.pth'使用0.001 Adam优化25轮，：63.0%

改用lr=0.0001优化

对于CIFAR100的每个小类的测试精确度，类resnet18卷积神经系统模型可在使用lr=0.01的SGD优化15轮后，再使用lr=0.0001的Adam优化4*4轮后可达74.8%；两层卷积层+GRU可达42%；三层卷积层+GRU可达61%；四层卷积层+GRU可达63.46%。

而对于大类

