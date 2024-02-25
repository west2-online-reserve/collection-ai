### Vision Transformer

![图](VIT.png)

将 Transformer 的思想应用到 CV 领域。

[原论文](https://arxiv.org/pdf/2010.11929.pdf)

思路简述：将图像分割成块，每个块视为一个单词应用进Transformer的模型中使用。

#### 数据预处理

将一张图片进行分割（假设输入数据形状 $224 \times 224$），原论文给出的大小为 $16 \times 16$，拼接入 cls 元素 (concatenate) 然后加入位置信息 (plus)，最后通过全连接层改变形状。

此处尝试将分割改为了一个卷积层

代码：
```py
class Pre_work(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, 784))
        self.emd = nn.Parameter(torch.randn(1, 197, 784))
        self.linear = nn.Linear(784, 768)
        self.covn = nn.Sequential(
            nn.Conv2d(3, 196, 8, 8),
            nn.Flatten(2),
            nn.Linear(784, 784)
        )
    
    def forward(self, x):
        x = self.covn(x)
        return self.linear(torch.cat([self.cls.expand(x.shape[0], 1, self.cls.shape[2]), x], dim = 1) + self.emd)

```

#### Transformer

Transformer是一种使用自注意力的 NLP 模型，此处只用到 Transformer 的 Encoder部分。

##### 注意力机制

注意力机制包含三个参数 ：query, value, key，value 和 key 应一一对应
注意力机制分为随意线索注意力和不随意线索注意力。
- 不随意线索：卷积，全连接，池化……
- 随意线索：$f(x) = \sum \frac {K(x - x_i)}{\sum{K(x - x_j)}} y_i$，其中 $x$ 是 query ， $x_i$ 是 $key$, $y_i$ 是 $value$

参数化的随即线索注意力机制：

$$f_x = \sum_i\text{softmax} (\alpha(x, x_i)) y_i$$

其中 $\alpha(x, x_i)$ 为注意力权重

此处 q, k, v 都是多个组成的序列，单个元素可以为向量。

注意力分数：

1. Additive Attention
    
    $$\alpha(q, k) = v^T\tanh(W_q q + W_k k)$$

    其中 $W_q$，$W_k$ 为可学习参数

2. Scaled Dot-Product Attention

    $$\alpha(q, k) = \frac{\langle q, k\rangle}{\sqrt d} $$

    此处要求 q, k 的长度相同，d 为长度。

自注意力：令 q, k, v 均等于给定的 x 即为对 x 的自注意力操作。

此处使用的注意力分数为 Scaled Dot-Product Attention

代码：

```py
class dotproductattention(nn.Module):
    def __init__(self, dropout = 0.15) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
    
    def forward(self, q, v, k):
        d = q.shape[-1]
        res = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)
        res = nn.functional.softmax(res, dim = -1)
        return torch.bmm(self.drop(res), v)
```

##### Multi-head Attention

对于同一 q, k, v 提取不同的信息，类似于卷积的多输出通道

假设 q, k, v 形状分别为 $d_q$, $d_k$, $d_v$, 第 i 个头的可学习参数为 $Wq_i, Wk_i, Wv_i$，形状分别为 $P_q \times d_q$，$P_k \times d_k$，$P_v \times d_v$， 其中 $P$ 为隐藏层大小。第i个头的输出为：

$$h_i = f(Wq_i q, Wk_i k, Wv_i v)$$

其中 f 表示注意力操作， 输出形状为 $P_v$。

假设输出形状为 $P_0$ 整体输出为 
$$W_o\begin{bmatrix}
h_0\\
h_1\\
\vdots\\
h_n
\end{bmatrix}$$

此处具体实现方式：

假设原维度为 $(b \ n \ d)$，将最后一维分裂，变为 $(b \ n\ (h \ d/h))$，然后将 h 前移，变为 $(b \ h \ n \ d)$，这样可以一次性将所有头的运算做完。最后在便会原先的形状。

代码：

```py
def trans_pos(x, n):
    x = x.view(x.shape[0], 197, n, -1)
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.reshape(-1, x.shape[2], x.shape[3])
    return x
def trans_neg(x, n):
    x = x.view(n, -1, x.shape[1], x.shape[2])
    x = x.permute(0, 2, 3, 1).contiguous()
    x = x.reshape(x.shape[0], x.shape[1], -1)
    return x

class multihead(nn.Module):
    def __init__(self, q_size, v_size, k_size, head_num, hide_num) -> None:
        super().__init__()
        self.attention = dotproductattention()
        self.W_q = nn.Linear(q_size, hide_num)
        self.W_v = nn.Linear(v_size, hide_num)
        self.W_k = nn.Linear(k_size, hide_num)
        self.W_o = nn.Linear(hide_num, q_size)
        self.head_num = head_num
        
    def forward(self, q, v, k):
        Q = trans_pos(self.W_q(q), self.head_num)
        K = trans_pos(self.W_k(k), self.head_num)
        V = trans_pos(self.W_v(v), self.head_num)
        res = self.attention(Q, V, K)
        # print("REs:", res.shape)
        return self.W_o(trans_neg(res, q.shape[0]))
```

#### Transformer 块

形式如上图所示，先进行归一化(LayerNorm), 然后进行多头自注意力，之后残差操作，再次归一化，全连接层，最后残差。此为一个 Transformer 块的内容。中间穿插使用dropout

代码：

```py
class vit_block(nn.Module):
    def __init__(self, n, dropout = 0.2) -> None:
        super().__init__()
        self.multihead = multihead(768, 768, 768, 12, 1536)
        self.l1 = nn.LayerNorm((197, 768))
        self.l2 = nn.LayerNorm((197, 768))
        self.l3 = nn.LayerNorm((197, 768))
        self.mlp = nn.Sequential(
            nn.Linear(768, n),
            nn.ReLU(),
            nn.Linear(n, 768)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        y = self.l1(x)
        y = self.multihead(y, y, y)
        y = self.dropout(y)
        y = y + x
        z = self.l2(y)
        z = self.mlp(z)
        z = self.dropout(z)
        z = z + y
        return self.l3(z)
```

#### 总体结构

先对图像进行分割图处理，然后进入多层的 Transformer，最后取 cls 对应是输出结果，通入全连接层改变形状

代码（参数为临时测试结果）：

```py
class vit(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pre = Pre_work()
        self.v1 = vit_block(2048, 0.)
        self.v2 = vit_block(2048, 0.)
        self.v3 = vit_block(2048, 0.)
        self.v4 = vit_block(2048, 0.1)
        self.v5 = vit_block(2048, 0.1)
        self.v6 = vit_block(2048, 0.55)
        self.linear = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 101)
        )
        self.drop = nn.Dropout(0.1)
        self.id = nn.Identity()
    
    def forward(self, x):
        x = self.pre(x)
        x = self.drop(x)
        x = self.v2(self.v1(x))
        x = self.v3(x)
        x = self.v4(x)
        x = self.v5(x)
        x = self.v6(x)
        x = self.id(x[:, 0, :])
        return self.linear(torch.squeeze(x))
```

训练时使用`optim.lr_scheduler.StepLR`进行学习率降低，参数为`optim.lr_scheduler.StepLR(optims, step_size = 1, gamma = 0.87)` 每 epoch 降低一次。初始学习率为 `1e-4`。

效果好的时候可在 10 个 epoch内达到最高正确率 55% 左右， loss 达到 2 左右。