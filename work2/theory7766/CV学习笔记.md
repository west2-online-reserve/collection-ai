##一、激活函数（给网络提供非线性的建模能力）
通过计算加权和并且加上偏置来确定神经元是否应该被激活，它们将输入信号转换为输出的可微运算。
###1.ReLU函数（修正线性单元）
ReLU(x)=max(x,0),它解决了神经网络中梯度消失的问题。它收敛快，计算简单。但较为脆弱，如遇到较大梯度经过ReLU单元，权重的更新结果可能是0，在此之后它将永远不能再被激活。

###2.sigmoid函数（挤压函数）
sigmoid(x)=1/(1+exp(-x)),将数据压缩到（0，1）之间，可被用作概率或输入的归一化，但易出现梯度消失。

 当我们想要将输出视作二元分类问题的概率时， sigmoid仍然被广泛用作输出单元上的激活函数 （sigmoid可以视为softmax的特例）。 然而，sigmoid在隐藏层中已经较少使用， 它在大部分时候被更简单、更容易训练的ReLU所取代。


###3.tanh函数（双曲正切函数）
tanh(x)=(1-exp(-2x))/(1+exp(-2x)),形状类似于sigmoid函数，但关于坐标原点中心对称，将数据控制在（-1，1），解决非0均值的问题，但它也存在梯度消失和梯度爆炸的问题


###4.LeakyReLU函数
LeakyReLU(x)={x,x>=0; γx,x<0
其中，γ是很小的负数梯度值。

LeakyReLU解决了一部分ReLU存在的可能杀死神经元的问题。它给所有非负值赋予一个非零的斜率，来保证负轴不为零，保证负轴信息的存在性，因此解决了一部分神经元死亡的问题。

##二、模型选择、欠拟合、过拟合
将模型在训练数据上拟合的比在潜在分布中更接近的现象称为过拟合（overfitting）， 用于对抗过拟合的技术称为正则化（regularization）。
###1.训练误差和泛化误差
训练误差（training error）是指， 模型在训练数据集上计算得到的误差。
 
泛化误差（generalization error）是指， 模型应用在同样从原始样本的分布中抽取的无限多数据样本时，模型误差的期望。

独立同分布假设：假设训练数据和测试数据都是从相同的分布中独立提取的。

影响模型泛化的因素：可调整参数的数量/参数采用的值/训练样本的数量（即使模型很简单，也很容易过拟合只包含一两个样本的数据集。而过拟合一个有数百万个样本的数据集则需要一个极其灵活的模型）

###2.模型选择

评估几个候选模型后选择最终的模型。通常使用验证集。

当训练数据稀缺时，使用K折交叉验证：原始训练数据被分成K个不重叠的子集。 然后执行K次模型训练和验证，每次在K-1个子集上进行训练， 并在剩余的一个子集（在该轮中没有用于训练的子集）上进行验证。 最后，通过对K次实验的结果取平均来估计训练和验证误差。

###3.过拟合还是欠拟合？

欠拟合：训练误差和验证误差都很严重， 但它们之间仅有一点差距。

过拟合：当我们的训练误差明显低于验证误差。

是否过拟合或欠拟合可能取决于模型复杂性和可用训练数据集的大小。

训练数据集中的样本越少，我们就越有可能（且更严重地）过拟合。随着训练数据量的增加，泛化误差通常会减小。

##三、权重衰减（L2正则化）
min l(w,b)+λ*||w||^2/2
其中，λ为非负超参数，是正则化常数。

岭回归算法：L2正则化线性模型（它对权重向量的大分量施加了巨大的惩罚，使得我们的学习算法偏向于在大量特征上均匀分布权重的模型）

套索回归算法：L1正则化线性回归
(L1惩罚会导致模型将权重集中在一小部分特征上， 而将其他权重清除为零。 这称为特征选择惩罚会导致模型将权重集中在一小部分特征上， 而将其他权重清除为零。 这称为特征选择)


	 trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)

##四、丢弃法dropout（通常用在多层感知机的隐藏全连接层的输出）
它将一些输出项随机置0。

丢弃概率是控制模型复杂度的超参数。
仅仅在训练时在层之间丢掉，影响模型参数更新，在推理中丢弃法直接返回输入。

	net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

##五、数值稳定性和模型初始化
###1.梯度爆炸和梯度消失
梯度爆炸问题： 参数更新过大，破坏了模型的稳定收敛； 

梯度消失问题： 参数更新过小，在每次更新时几乎不会移动，导致模型无法学习。

我们在每一层的隐藏单元之间具有排列对称性。虽然小批量随机梯度下降不会打破这种对称性，但暂退法正则化可以。


###2.让训练更加稳定

乘法变加法：ResNet，LSTM

归一化：梯度归一化，梯度裁剪

合理的权重初始和激活函数

###3.参数初始化

1）默认初始化：默认的随机初始化

2）Xavier初始化：高斯分布（0，2/（n_in+n_out) 均匀分布U(-根号（6/（n_in+n_out)），根号（6/（n_in+n_out)））

随机初始化是保证在进行优化前打破对称性的关键。

Xavier初始化表明，对于每一层，输出的方差不受输入数量的影响，任何梯度的方差不受输出数量的影响。
##六、深度学习计算
###1.层和块
块由类（class）表示。 它的任何子类都必须定义一个将其输入转换为输出的前向传播函数， 并且必须存储任何必需的参数。

自定义块

	class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))

顺序块

	class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
###2.参数管理
获取第一层全部参数：
	
	net[0].state_dict()

net[0].bias.data(偏移数据）

获取全部参数：

	print(*[(name, param.shape) for name, param in net.named_parameters()])

内置初始化

	def xavier(m):
		if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

参数绑定：使用稠密层，使用其参数来设置另一层的参数

		# 我们需要给共享层一个名称，以便可以引用它的参数
	shared = nn.Linear(8, 8)
	net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))

###3.自定义层
带参数的自定义层

	class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])

###4.读写文件

加载和保存张量x：
torch.save(x,'x-file')  torch.load(x,'x-file')

加载和保存模型参数：
torch.save(net.state_dict(), 'mlp.params')

	clone=MLP()
	#把参数加载到模型
	clone=load_state_dict(torch.load('mlp.params'))
	clone.eval()#进入测试模式

##七、卷积神经网络CNN
###1.从全连接层到卷积层

1.平移不变性:
二维卷积(交叉相关） h_i,j=Σ V_a,b*X_i+a,j+b

2.局部性:
当|a|,|b|>Δ，使得V_a,b=0
###2.卷积层

将输入与核矩阵进行交叉·相关，加上偏移以后得到输出。
其中，核矩阵与偏移是可学习的参数。核矩阵的大小是超参数。
输入X:n_h×n_w

核W:k_h×k_w

偏差b属于R

输出Y:(n_h-k_h+1)×(n_w-k_w+1)

Y=X*W+b

###3.填充和步幅（卷积层超参数）

填充padding：在输入图像的边界填充元素（通常填充元素是0）

填充后，输出形状为
(n_h-k_h+p_h+1)×(n_w-k_w+p_w+1)

通常，p_h=k_h-1,p_w=k_w-1

步幅stride：每次滑动元素的数量
输出形状为(n_h-k_h+p_h+1)/s_h×(n_w-k_w+p_w+1)/s_w
	
	conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)

###4.多输入多输出通道

输出通道数是卷积层的超参数。

针对每个通道，都有相应的二维卷积核，通道结果相加得到输出通道结果。

每个输出通道可以识别特定模式，都有独立的三维卷积核。

输入通道核识别并且组合输入中的模式。

1×1卷积层：k_h=k_w=1,不识别空间模式，仅为融合通道。 

###5.池化层/汇聚层
目的：降低卷积层对位置的敏感性，同时降低对空间降采样表示的敏感性。

二维最大池化pooling,返回滑动窗口中的最大值。

与卷积层类似，均有填充和步幅，但其无可学习的参数。在每个输入通道应用池化层来获得相应的输出通道。

输出通道数=输入通道数

平均池化层：返回滑动窗口中的平均值。

默认情况下，深度学习框架中的步幅与汇聚窗口的大小相同。

	pool2d = nn.MaxPool2d(3, padding=1, stride=2)
	#stride(2,3)先2垂直滑动，再3水平滑动
	pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))

###6.经典卷积神经网络LeNet

组成部分：
1.卷积编码器：由2个卷积层组成

2.全连接层密集块：由三个全连接层组成（使用其转换到类别空间）

##八、现代卷积神经网络

1.深度卷积神经网络AlexNet

它是第一个在大规模视觉竞赛中击败传统计算机视觉模型的大型神经网络；

主要改进：丢弃法、ReLU、MaxPooling


2.残差网络（ResNet）。它通过残差块构建跨层的数据通道，是计算机视觉中最流行的体系架构；

3.使用重复块的网络（VGG）。它利用许多重复的神经网络块；

4.网络中的网络（NiN）。它重复使用由卷积层和1*1卷积层（用来代替全连接层）来构建深层网络;

5.含并行连结的网络（GoogLeNet）。它使用并行连结的网络，通过不同窗口大小的卷积层和最大汇聚层来并行抽取信息；

6.稠密连接网络（DenseNet）。它的计算成本很高，但给我们带来了更好的效果。

##九、技术名词
###1.网络层
1.1Conv2D:实现2D卷积

in_channels —— 输入的channels数，
out_channels —— 输出的channels数，
kernel_size ——卷积核的尺寸，
stride —— 步长，
padding ——输入边沿扩边操作，
padding_mode ——扩边的方式，
bias ——是否使用偏置(即out = wx+b中的b)
dilation——取数之间的间隔,
groups —— 进行分组卷积的组数

1）方形卷积核、行列相同步长（With square kernels and equal stride）

m = nn.Conv2d(16, 33, 3, stride=2)

2）非方形卷积核、行列采用不同步长，并进行扩边

	m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))

3）非方形卷积核、行列采用不同步长、数据采用稀疏，并进行扩边

	m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
1.2.Conv1D:对一维数据进行卷积，常用于处理序列数据，如文本、音频和时间序列。

1.3.BatchNormalization:批量归一化，是正则化技术之一。

	torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, 
	affine=True, track_running_stats=True, device=None, dtype=None) 

1.num_features：特征的数量

2.eps：分母中添加的一个常数，为了计算的稳定性，默认为 1e-5；

3.momentum：用于计算 running_mean 和 running_var，默认为 0.1；

4.affine：当为 True 时，会给定可以学习的系数矩阵 gamma 和 beta；

5.track_running_stats：一个布尔值，当设置为 True 时，模型追踪 running_mean 和 running_variance，当设置为 False 时，模型不跟踪统计信息，并在训练和测试时都使用测试数据的均值和方差来代替，默认为 True。


1.4LayerNormalization:LN 针对单个训练样本进行，不依赖于其他数据，因此可以避免 BN 中受 mini-batch 数据分布影响的问题，可以用于 小 mini-batch 场景、动态网络场景和 RNN，特别是自然语言处理领域。此外，LN 不需要保存 mini-batch 的均值和方差，节省了额外的存储空间。

1.5Dropout丢弃法，为正则化手段之一参见上面的介绍

1.6LSTM（长短期记忆网络）

它是循环神经网络RNN的特例，可避免常规RNN的梯度消失，解决常依赖问题。

1.7GRU（门控循环单元）

它是循环神经网络RNN的变种，相较于LSTM，它更为简单。包含更新门和重置门。

###2.损失函数
2.1 CrossEntropy:交叉熵损失函数

2.2 MSE：均方方差，预测数据和原始数据对应点误差的平方和的均值，大多数变量都可以建模为高斯分布

2.3 MAE：平均绝对误差，目标值和预测值之差的绝对值之和

2.4 Logcosh:预测误差的双曲余弦的对数

###3.优化方法

3.1 RMSprop：在Adagrad基础上将原来的梯度除以方差的平方根来更新参数。

RMSProp主要思想：使用指数加权移动平均的方法计算累积梯度，以丢弃遥远的梯度历史信息（让距离当前越远的梯度的缩减学习率的权重越小）。
3.2 SGD：随机梯度下降，易陷入局部最优解
3.3 Adagrad：独立地适应所有模型参数的学习率，缩放每个参数反比于其所有梯度历史平均值总和的平方根。在每次迭代中，会根据之前的梯度信息自动调整每个参数的学习率。
3.4 Adadelta：避免使用手动调整学习率的方法来控制训练过程，而是自动调整学习率，使得训练过程更加顺畅。
3.5 Adam：Momentum&&RMSprop结合体
3.6 Nadam ：Adam&&NAG

###4.优秀网络
4.1 Vit:ViT将输入图片分为多个patch（16 * 16）,再将每个patch投影为固定长度的向量送入Transformer，后续encoder的操作和原始Transformer中完全相同。
4.2Yolo:一次性预测多个Box位置和类别的卷积神经网络能够实现端到端的目标检测和识别，其最大的优势就是速度快。