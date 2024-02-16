### 思路步骤

1、数据集加载

2、将数据由二维铺开为一维

3、设置神经网络

4、神经网络训练

5、用训练好的模型预测，测试准确率



参考文章：原文链接：https://blog.csdn.net/m0_62237233/article/details/127153475

#### 导入数据



使用Pytorch自带数据库**torchvision.datasets，**通过代码下载torchvision.datasets中的MNIST数据集



**函数原型：**

```python
torchvision.datasets.MNIST(root,
                           train=True, 
                           transform=None, 
                           target_transform=None, 
                           download=False)
```



参数说明：

root (string) ：数据地址
train (string) ：True = 训练集，False = 测试集
download (bool,optional) : 如果为True，从互联网上下载数据集，并把数据集放在root目录下。
transform (callable, optional )：这里的参数选择一个你想要的数据转化函数，直接完成数据转化
target_transform (callable,optional) ：接受目标并对其进行转换的函数/转换。





训练集：

```python
train_ds = torchvision.datasets.MNIST('data', 
train=True, 
transform=torchvision.transforms.ToTensor(), 
download=True)
```





测试集

```python
test_ds = torchvision.datasets.MNIST('data', 
train=True, 
transform=torchvision.transforms.ToTensor(), 
download=True)
```





###### 补充：tensor数据

tensor——张量

常见的数据类型，一个多维数组，可以直接再GPU上运算‘

在深度学习上很有用



使用Pytorch自带的数据加载器**torch.utils.data.DataLoader，**结合了数据集和采样器，并且可以多线程处理数据集。



**函数原型：**

```python
torch.utils.data.DataLoader(dataset,
                            batch_size=1, 
                            shuffle=None,
                            sampler=None,
                            batch_sampler=None, 
                            num_workers=0, 
                            collate_fn=None,
                            pin_memory=False,
                            drop_last=False,
                            timeout=0, 
                            worker_init_fn=None,
                            multiprocessing_context=None,
                            generator=None,
                            *, 
                            prefetch_factor=2, 
                            persistent_workers=False,
                            pin_memory_device='')

```

参数说明：

**dataset(string) ：加载的数据集（从哪读取）**
**batch_size (int,optional) ：每批加载的样本大小（默认值：1）**
**shuffle(bool,optional) : 如果为True，每个epoch重新排列数据。（是否乱序）**
sampler (Sampler or iterable, optional) ： 定义从数据集中抽取样本的策略。 可以是任何实现了 __len__ 的 Iterable。 如果指定，则不得指定 shuffle 。
batch_sampler (Sampler or iterable, optional) ： 类似于sampler，但一次返回一批索引。与 batch_size、shuffle、sampler 和 drop_last 互斥。
**num_workers(int,optional) ： 用于数据加载的子进程数。 0 表示数据将在主进程中加载（默认值：0）。（是否多进程）**
pin_memory (bool,optional) : 如果为 True，数据加载器将在返回之前将张量复制到设备/CUDA 固定内存中。 如果数据元素是自定义类型，或者collate_fn返回一个自定义类型的批次。
drop_last(bool,optional) : 如果数据集大小不能被批次大小整除，则设置为 True 以删除最后一个不完整的批次。 如果 False 并且数据集的大小不能被批大小整除，则最后一批将保留。 （默认值：False）
timeout(numeric,optional) : 设置数据读取的超时时间 ， 超过这个时间还没读取到数据的话就会报错。（默认值：0）
worker_init_fn(callable,optional) ： 如果不是 None，这将在步长之后和数据加载之前在每个工作子进程上调用，并使用工作 id（[0，num_workers - 1] 中的一个 int）的顺序逐个导入。 （默认：None）



备注：大多数没看懂怎么使用，等有用的再了解。加粗为常用。



加载数据：

```python
batch_size = 32
 
 # 训练集
train_dl = torch.utils.data.DataLoader(train_ds, 
                                       batch_size=batch_size, 
                                       shuffle=True)
 # 测试集
test_dl  = torch.utils.data.DataLoader(test_ds, 
                                       batch_size=batch_size)
```



测试： 

```python
# 取一个批次查看数据格式
# 数据的shape为：[batch_size, channel, height, weight]
# 其中batch_size为自己设定，channel，height和weight分别是图片的通道数，高度和宽度。
imgs, labels = next(iter(train_dl))
print(imgs.shape)
```





#### 数据可视化

**squeeze（）函数**：将矩阵shape中维度为1的去除，比如一个矩阵的shape为（3，1），通过此函数为（3，）。



也即上述步骤中将二维数组铺开为一维数组





#### 搭建网络



###### 常用函数

 nn.Conv2d为[卷积层](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)，用于提取图片的特征，传入参数为输入channel，输出channel，池化核大小

```python
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

```

输入图片的矩阵 *I* 的大小：w*w； 卷积核K：k*k； 步长S：s；   填充大小（padding）：p

有o=（w-k+2*p）/s + 1；   输出图片大小即为o*o



nn.MaxPool2d为[池化层](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d)，进行下采样，用更高层的抽象表示图像特征，传入参数为池化核大小

```python
torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)


```

kernel_size：最大的窗口大小；  stride：窗口的步幅，默认值为kernel_size；   

padding： 填充大小（默认0）；dilation：控制窗口中元素步幅的参数，默认为1

如代码：input_x = torch.randn(3, 32, 32)    //具体运算公式见文档

输入的数据格式是：[3, 32, 32]即 [C*Hin*Win]



nn.ReLU为[激活函数](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html?highlight=relu#torch.nn.ReLU)，使模型可以拟合非线性数据。输入：任意数量的维度，输出：与输入的形状相同。



nn.Linear为[全连接层](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear)，对输入数据应用线性变换。参数（每个输入样本大小，每个输出样本大小）



nn.Sequential可以[按构造顺序连接网络](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html?highlight=sequential#torch.nn.Sequential)，在初始化阶段就设定好网络结构，不需要在前向传播中重新写一遍。是pytorch中的顺序容器，构建可以直接嵌套或者以orderdict有序字典的方式传入。







### 自己代码：

```python
transform = transforms.Compose([
    transforms.ToTensor(),  # 可将PIL类型的图像或者Numpy数组转为Tensor，并将其像素值归一化到[0,1]
])


# 训练集、测试集
train_dataset = datasets.MNIST(
    root='./dataset',
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root='./dataset',
    train=False,
    transform=transform,
    download=True
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    drop_last=False
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    drop_last=False
)
```

准备训练集和测试集



```
class DemoModel(nn.Module):
    def __init__(self):
        super(DemoModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding='same'),
            nn.AvgPool2d(kernel_size=2),  # （32,1,14,14）
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=14 * 14 * 16, out_features=10),
            nn.Softmax()
        )

    def forward(self, x):
        return self.model(x)
```

构建神经网络的模型

分析该模型作用：

先继承，然后调用nn.Sequential()来写步骤

“

nn.Sequential:一个序列容器，用于搭建神经网络的模块被按照被传入构造器的顺序添加到nn.Sequential()容器中。

###### 本质作用

与一层一层的单独调用模块组成序列相比，nn.Sequential() 可以允许将整个容器视为单个模块（即相当于把多个模块封装成一个模块），forward()方法接收输入之后，nn.Sequential()按照内部模块的顺序自动依次计算并输出结果。这也是为啥要先继承

”

分析nn.Sequential容器里每一步作用：

1、nn.Conv2d为[卷积层](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)，用于提取图片的特征，in_channels：输入的通道数（因为灰度图片，所以为1，若为RBG，则为3）， out_channels：输出的通道数，kernel_size：卷积核的大小（本质为元组，但输入一般为 5x5、3x3 ，所以简写3，若遇到5x3，则为（5，3）），padding：指图像填充，这里是手写数字识别，所以用不到空洞卷积和分组卷积，虽然我也不会

2、nn.AvgPool2d——二维平均[池化](https://so.csdn.net/so/search?q=池化&spm=1001.2101.3001.7020)操作，在由多个平面组成的输入信号上应用2D平均池化操作

![](D:\python\机器学习\3\10.png)

按我自己人话理解池化就是在干一件事：将图片大小缩放但不改变图片的特征值，所以我们使用一次平均池化将原本卷积出来的图片进行缩放到我们需要的尺寸

3、nn.BatchNorm2d——批量标准化操作，功能：对输入的四维数组进行批量标准化处理，具体计算公式如下：

![](D:\python\机器学习\3\11.png)

对于**所有的batch**中样本的**同一个channel**的数据元素进行标准化处理，即如果有C个通道，无论batch中有多少个样本，都会在通道维度上进行标准化处理，一共进行C次。

**num_features**：输入图像的通道数量-C。

4、nn.ReLU为[激活函数](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html?highlight=relu#torch.nn.ReLU)，使模型可以拟合非线性数据。输入：任意数量的维度，输出：与输入的形状相同。

神经网络每一次向前都需要激活函数，这次代码把两次卷积的步骤合在一起写了，应该可以跟规范一些

5、再次进行卷积的操作来提取一次特征值，具体可以看下图说明：

![](D:\python\机器学习\3\12.jpg)

想要识别手写数字需要两次的卷积来提取特征值，至于为什么我还没学明白，先放在这里

由于第一次从一个chennel变到八个，，所以本次输入通道为八，接下来的步骤与第一次卷积相同，依旧会有一个激活函数



6、**torch.nn.Flatten(start_dim=1, end_dim=- 1)**

作用：将连续的维度范围展平为张量。 经常在nn.Sequential()中出现，一般写在某个神经网络模型之后，用于对神经网络模型的输出进行处理，得到tensor类型的数据。有俩个参数，start_dim和end_dim，分别表示开始的维度和终止的维度，默认值分别是1和-1，其中1表示第一维度，-1表示最后的维度。结合起来看意思就是从第一维度到最后一个维度全部给展平为张量。

但实际上数据是由第零个维度的，但是好像我上网搜的都是从第一个维度开始处理，都没有用到第零个维度，所以我感觉第零维度就是偏置函数，它实际上对神经网络整体的运算是没有影响的，所以不需要去考虑它

7、nn.Linear为[全连接层](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear)，对输入数据应用线性变换。

nn.Linear定义一个神经网络的线性层，方法签名如下：

```python
torch.nn.Linear(in_features, # 输入的神经元个数
           out_features, # 输出神经元个数
           bias=True # 是否包含偏置
           )
```

![](D:\python\机器学习\3\13.png)

这里的b就是偏置值，这其实就是神经网络的反向传播，和向前传播时相对的

8、nn.Softmax() 老熟人了，激活函数，归一化处理，得到每一个值的概率，也就是我们输入的图片经过神经网络的卷积和学习，最后给出了它结果的概率

以上就是我对于手写数字识别的神经网络构建的理解，虽然很多东西还是借鉴网络上的，也还有很多还没搞懂，只能慢慢来解决了



```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 确定采用cpu还是gpu计算
model = DemoModel()
model = model.to(device)  # gpu加速
creation = nn.CrossEntropyLoss()  # 损失函数：分类任务常用交叉熵函数，二分类任务可用nn.BCELoss()
creation = creation.to(device)

# 5.优化器
learning_rate = 1e-2
beta1 = 0.9
beta2 = 0.999
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2))  # 优化器
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32)  # 学习率衰减：余弦退火策略
```

这里有两个函数我需要学习：

1、optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2))

优化器定义函数（参考：[Python-torch.optim优化算法理解之optim.Adam()_静静喜欢大白的博客-CSDN博客](https://blog.csdn.net/lj2048/article/details/114889359?ops_request_misc=%7B%22request%5Fid%22%3A%22170185377916800192292613%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=170185377916800192292613&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-114889359-null-null.142^v96^control&utm_term=torch.optim.Adam&spm=1018.2226.3001.4187)）

[torch](https://so.csdn.net/so/search?q=torch&spm=1001.2101.3001.7020).optim是一个实现了多种优化算法的包，大多数通用的方法都已支持，提供了丰富的接口调用，未来更多精炼的优化算法也将整合进来。为了使用torch.optim，需先**构造一个优化器对象Optimizer**，用来保存当前的状态，并能够根据计算得到的梯度来更新参数。

使用步骤：

```python
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

Adam(Adaptive Moment Estimation)本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。它的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。其公式如下：

![](D:\python\机器学习\3\14.png)

（嗯，看不懂，乐）

其中，前两个公式分别是对梯度的一阶矩估计和二阶矩估计，可以看作是对期望E|gt|，E|gt^2|的估计;
公式3，4是对一阶二阶矩估计的校正，这样可以近似为对期望的无偏估计。可以看出，直接对梯度的矩估计对内存没有额外的要求，而且可以根据梯度进行动态调整。

最后一项前面部分是对学习率n形成的一个动态约束，而且有明确的范围。

lr：同样也称为学习率或步长因子，它控制了权重的更新比率（如 0.001）。较大的值（如 0.3）在学习率更新前会有更快的初始学习，而较小的值（如 1.0E-5）会令训练收敛到更好的性能。
betas = （beta1，beta2）
beta1：一阶矩估计的指数衰减率（如 0.9）。
beta2：二阶矩估计的指数衰减率（如 0.999）。该超参数在稀疏梯度（如在 NLP 或计算机视觉任务中）中应该设置为接近 1 的数。
eps：epsilon：该参数是非常小的数，其为了防止在实现中除以零（如 10E-8）。

```python
# 6.网络训练
model_save_path = './models/'
os.makedirs(model_save_path, exist_ok=True)

EPOCHS = 50
train_steps = 0
test_steps = 0
for epoch in range(EPOCHS):  # 0-99
    print("第{}轮训练过程：".format(epoch + 1))
    # 训练步骤
    model.train()  # 训练模式
    epoch_train_loss = 0.0
    for train_batch, (train_image, train_label) in enumerate(train_dataloader):
        train_image, train_label = train_image.to(device), train_label.to(device)  # 将数据送到GPU
        train_predictions = model(train_image)
        batch_train_loss = creation(train_predictions, train_label)
        optimizer.zero_grad()  # 梯度清零
        batch_train_loss.backward()
        optimizer.step()
        epoch_train_loss += batch_train_loss.item()
        train_steps += 1

    # 测试步骤
    model.eval()
    epoch_test_loss = 0.0
    epoch_test_acc = 0.0
    with torch.no_grad():
        for test_batch, (test_image, test_label) in enumerate(test_dataloader):
            test_image, test_label = test_image.to(device), test_label.to(device)
            predictions = model(test_image)
            test_loss = creation(predictions, test_label)
            epoch_test_loss += test_loss.item()
            test_steps += 1
            # 计算每个批次数据测试时的准确率
            batch_test_acc = (predictions.argmax(dim=1) == test_label).sum()
            epoch_test_acc += batch_test_acc

    if (epoch + 1) % 10 == 0:
        print("第{}轮训练结束，训练损失为{}，测试损失为{}，测试准确率为{}".format
              (epoch + 1, epoch_train_loss, epoch_test_loss, epoch_test_acc / len(test_dataset)))
             
# 7.保存模型
    torch.save(model.state_dict(), model_save_path + "model{}.pth".format(epoch + 1))
print("训练结束！")
```

这里的训练步骤其实就是上述optimizer的使用方法，我是根据此使用方法来写这里训练步骤的，然后规定了训练一共50次，使得准确率比较高，同时在测试的时候加入损失函数的计算，得到损失率和准确率

最后保存每一次的到的模型

