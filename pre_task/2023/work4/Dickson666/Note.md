# Yolo v1

通过darknet提取特征图，大小为 7 * 7 * 30， 7 * 7 为特征图大小，30 为 5 + 5 的预测框 和 20 的分类

### 思路

特征图上每一个位置对应一个物体（物体中心落在该格子内），这个位置负责预测该物体。同时每个位置只预测一个物体，这也是YOLO v1的缺陷

### loss
$$
\begin{split}
\text{Loss} &= \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} [(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2] \\
&+ \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} [( \sqrt{w_i} - \sqrt{\hat{w}_i})^2 + ( \sqrt{h_i} - \sqrt{\hat{h}_i})^2] \\
&+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2 \\
&+ \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2 \\
&+ \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2
\end{split}
$$

- $S$ 是特征图大小，此处为 7；
- $B$ 是每个位置预测的框的个数，此处为 2；
- $1_{ij}^{obj}$ 表示 $i$ 位置的 $j$ 框中有无物体，若有则为 1，否则为 0，$1_{ij}^{noobj}$ 与之相反；

第一、二行为 boundingbox 的 loss，在长宽处取根号是为了平衡大框与小框的 loss（大框一个单位的偏移会产生较小的 loss，但小框同样一个单位偏移应该产生更多 loss）。 $\lambda_{coord}$ 为系数，此处为 5；
第三、四行是置信度的 loss，有物体的位置 $C_i$ 为 IoU（好像也可以为 1），没有物体的则为 0。由于没有物体的位置会更多一些，引入了参数 $\lambda_{noobj}$ 用于平衡 loss，此处为 0.5；
最后一行为分类的 loss，每个位置的所有框共用分类。

### 网络模型

![](1.jpg)

原论文中给出的模型，不含 BatchNorm 层，最后为 flatten 后两层全连接最后再 reshape。

预测框的计算规则为（经过归一化）:

$b_x = \sigma(t_x) + c_x \\
 b_y = \sigma(t_y) + c_y \\
 b_w = \sigma(t_w)\\
 b_h = \sigma(t_h)$ 

# Yolo v2

对 v1 进行了一定程度的改进。

### BatchNorm

在网络中加入的 BatchNorm 层

### 更高的预训练分辨率

预训练时，先在 224 * 224 的图像上训练，最后 10 个 epoch 转到大小为 448 * 448 的分辨率上进行训练

### 加入先验框

类似于 SSD 加入先验框，但其先验框是在数据集中做 k-means，每个 box 与聚类中心 box 的距离指标为 1 - IOU。此处取 k = 5

### 新的网络

使用 Darknet19，不含全连接层，从而解决了全连接层会破环特征图形状的问题, 同时网络输出形状修改为 S, S, B * (5 + C)。由于先验框的加入，预测框的计算中 $b_w, b_h$ 发生变化：

$b_w = p_we^{t_w}\\
 b_h = p_he^{t_h}$

 ### passthrough 层

 引入 passthrough 层，即将网络中部大小为26 * 26 * 512 的特征图提取出来变化为 13 * 13 * 2048 的特征图拼接在网络的输出层上，形成 13 * 13 * 3072 的特征图。
 ![](3.png)

 ### 多分辨率训练

 由于网络中不含有全连接层，可以适应任意分辨率的输入，训练时采用 320 - 608 中 32 的倍数为分辨率，每 10 个 iterations 随机选择一种分辨率进行训练

 ### 损失函数

 分类和置信度处与 v1 的损失函数差别不大，预测框的置信度变为

 $$\begin{split}
 Loss_{box} &= \lambda_{coord} \sum_{i=0}^{S^2}\sum_{j=0}^B 1_{ij}^{obj}(2 - w_i h_i)[(x_i - \hat{x}_i) ^ 2 + (y_i - \hat{y}_i) ^ 2 + (w_i - \hat{w}_i) ^ 2 + (h_i - \hat{h}_i) ^ 2]\\
  &+ 0.01\sum_{i = 0}^{S^2}\sum_{j = 0}^B 1_{ij}^{noobj}[(p_{jx}-\hat{x}_i)^2 + (p_{jy}-\hat{y}_i)^2 + (p_{jw}-\hat{w}_i)^2 + (p_{jh}-\hat{h}_i)^2]\\
  \end{split}$$

  其中第二行仅在前 12800 个样本中计算

  