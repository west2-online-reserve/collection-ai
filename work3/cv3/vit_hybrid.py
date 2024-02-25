from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=[1, 1, 1],
        padding=[0, 1, 0],
        first=False,
    ) -> None:
        """

        :param in_channels:
        :param out_channels:
        """
        super(ResBlk, self).__init__()
        self.bottleneck = nn.Sequential(
            StdConv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride[0],
                padding=padding[0],
                bias=False,
            ),
            nn.GroupNorm(8,out_channels),
            nn.ReLU(inplace=True),
            StdConv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride[1],
                padding=padding[1],
                bias=False,
            ),
            nn.GroupNorm(8,out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels * 4,
                kernel_size=1,
                stride=stride[2],
                padding=padding[2],
                bias=False,
            ),
            nn.GroupNorm(8,out_channels * 4),
        )
        self.shortcut = nn.Sequential()
        if first:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * 4,
                    kernel_size=1,
                    stride=stride[1],
                    bias=False,
                ),
                nn.GroupNorm(8,out_channels * 4),
            )

    def forward(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet50(nn.Module):
    def __init__(self, ResBlk, num_classes=20):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            StdConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8,64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # conv2
        self.conv2 = self._make_layer(ResBlk, 64, [[1, 1, 1]] * 3, [[0, 1, 0]] * 3)

        # conv3
        self.conv3 = self._make_layer(
            ResBlk, 128, [[1, 2, 1]] + [[1, 1, 1]] * 3, [[0, 1, 0]] * 4
        )

        # conv4
        self.conv4 = self._make_layer(
            ResBlk, 256, [[1, 2, 1]] + [[1, 1, 1]] * 8, [[0, 1, 0]] * 9
        )



    def _make_layer(self, block, out_channels, strides, paddings):
        layers = []
        # 用来判断是否为每个block层的第一层
        flag = True
        for i in range(0, len(strides)):
            layers.append(
                block(
                    self.in_channels, out_channels, strides[i], paddings[i], first=flag
                )
            )
            flag = False
            self.in_channels = out_channels * 4
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        return out

def drop_path(x, drop_prob: float = 0., training: bool = False):


    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class StdConv2d(nn.Conv2d):

    def __init__(
            self, in_channel, out_channels, kernel_size, stride=1, padding=None,
            dilation=1, groups=1, bias=False, eps=1e-6):

        super().__init__(
            in_channel, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.eps = eps

    def forward(self, x):
        weight = F.batch_norm(
            self.weight.reshape(1, self.out_channels, -1), None, None,
            training=True, momentum=0., eps=self.eps).reshape_as(self.weight)
        x = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def _init_vit_weights(m):

    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
class PatchEmbed(nn.Module):

    # 这里的图片规格是224*224*3,图像分割每份大小是16*16，三通道RGB图像，再vit-B/16中的默认的dim是768
    # def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
    #     super().__init__()
    #     img_size = (img_size, img_size)#先把卷积核大小改为16*16
    #     patch_size = (patch_size, patch_size)
    #     self.img_size = img_size
    #     self.patch_size = patch_size
    #     self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1]) # 得出的结果是14*14,图的个数是14*14
    #     self.num_patches = self.grid_size[0] * self.grid_size[1]#patches的数目是14*14
    #     #这里把前面的参数丢进卷积里了（3,768,16*16,16）
    #     self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
    #     #如果norm_layer=None的话，就会初始化一个norm_layer,传入的话就用传入的,没传入就用nn.Identity()
    #     self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def __init__(self, img_size=14, patch_size=14, in_c=1024, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)#先把卷积核大小改为16*16
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1]) # 得出的结果是14*14,图的个数是14*14
        self.num_patches = self.grid_size[0] * self.grid_size[1]#patches的数目是14*14
        #这里把前面的参数丢进卷积里了（3,768,16*16,16）
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        #如果norm_layer=None的话，就会初始化一个norm_layer,传入的话就用传入的,没传入就用nn.Identity()
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    #前向传播，正式将图片数据传入进来
    def forward(self, x):
        B, C, H, W = x.shape
        #如果图片传入的高和宽不相等就会报错，因为在这个模型中输入的高和宽都是固定的
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW],进行一个展平处理
        # transpose: [B, C, HW] -> [B, HW, C],进行把1,2维进行一个调换
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

# 其实这里就是Mutihead-Self-Attention
class Attention(nn.Module):
    def __init__(self,
                 dim,   #向量的长度，就是token_dim，记得就是需要加一的那个
                 num_heads=8,#这个是Multi-Head Attention 的头的个数
                 qkv_bias=False,#是否使用qkv的偏置,默认false
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        #先赋值
        self.num_heads = num_heads
        #计算每一个head的dim，每个head都是被拆分过的，qkv就会被均分成多少份，对于每一个head的qkv就是dim除以划分的头的个数，就是dim除以头的个数
        head_dim = dim // num_heads
        #如果传入qk_scale,就用它的，如果没有传入,就自己算一个，这个根号下负的dim分之一，具体看文章给的Attention(Q,K,V)公式
        self.scale = head_dim ** -0.5#这个东西就是根号下d k分之一
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)#qkv是通过一个全连接层实现的。其实应该用三个全连接层，但是这里的节点个数有*3，效果就是一样的，可能这样写是提高运行速度
        self.attn_drop = nn.Dropout(attn_drop_ratio)#定义dropout层，用的就是上面的参数
        self.proj = nn.Linear(dim, dim)#头和W的映射，通过全连接层实现
        self.proj_drop = nn.Dropout(proj_drop_ratio)#定义dropout层，用的就是上面的参数

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        #这里的3就是qkv给拆分成3部分，num_heads就是头的个数，C // self.num_heads就是每一个heads 的qkv对应的个数
        #permute调整数据的顺序
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        #经过reshape,qkv的3被放在了最前面,这个时候就可以通过切片取出qkv,分别对应的是0,1,2
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale #这里是q乘k的转置，因为已经把qkv数据进行划分，这里就是依照每个head的qkv进行操作，运算符@是张量相乘（矩阵相乘），其实只计算了最后两个维度，最后两个转置了才能矩阵相乘
        #上一步已经完成了对于Attention(Q,K,V)公式里softmax括号里的公式的复现
        attn = attn.softmax(dim=-1)#这里-1就是对每一行都进行softmax处理
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        #现在是在softmax之后对每个v进行加权求和，就是完成Attention(Q,K,V)公式的最后一步
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)# 先进行矩阵相乘，运算符@是张量相乘（矩阵相乘），再进行维度处理，再将最后的head结果进行concat拼接（就是把b1,1两个dim  b1,2 合并成b1,1b1,2四个dim），reshape就完成了拼接

        x = self.proj(x)#通过W进行映射
        x = self.proj_drop(x)#通过dropout进行输出
        return x

class Mlp(nn.Module):

    #hidden_features=None 是第一个全连接层的节点个数，一般是输入的4倍,out_features和in_features是一样的
    def __init__(self, in_features, hidden_features=None, drop=0.):
        super().__init__()
        #如果有传入，就用传入值，没传入就用默认的
        out_features = in_features
        hidden_features = hidden_features or in_features
        #两个全连接层
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim, # 向量的长度，就是token_dim，记得就是需要加一的那个
                 num_heads, # 这个是Multi-Head Attention 的头的个数
                 mlp_ratio=4., # 这个是Mlp Block层翻的倍数，一般是4倍
                 qkv_bias=False, # 是否使用qkv的偏置
                 drop_ratio=0., # 不同的dropout层的参数
                 attn_drop_ratio=0.,
                 drop_path_ratio=0., # 这个用的是drop_path
                 act_layer=nn.GELU, # 激活函数是GELU
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)#对应的就是图里第一个layerNorm
        #接下来就调用Attention这个模块
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=False,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()#同样和图里的一样
        #图里的norm_layer
        self.norm2 = norm_layer(dim)
        #图里的mlp
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,  drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) # 通过norm1,attn, drop_path最后加上x，完成第一部分
        x = x + self.drop_path(self.mlp(self.norm2(x))) # 通过norm2，mlp，drop_path最后加上x，完成第二部分
        return x

#开始定义Vit类
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                  drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):

        super(VisionTransformer, self).__init__()

        #先赋值
        self.R50=ResNet50(ResBlk)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens =  1#这里num_token=1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)#这里的norm_layer是nn.LayerNorm，通过partial传入默认参数eps
        act_layer = act_layer or nn.GELU
        # 构建Patch Embedding
        self.patch_embed = embed_layer(img_size=14, patch_size=1, in_c=1024, embed_dim=768)
        num_patches = self.patch_embed.num_patches #获得patches的总个769

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))#先初始化了一个矩阵，就是那个1，第一个维度的1是用于后面的拼接，其实这里的token就是1*768

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))#初始化PositionEmbedding,patches是14*14=196,tokens是1,加起来就是197
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule，构建了一个等差数列，在每一个Encoder Block重复的次数里，drop_path都是递增的
        # 写了一个for循环，一共生成depth次Block Embedding,Sequential把depth个Block打包成一个block
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)#构建一个Block Embedding后面的那个layer Norm看原图

        self.has_logits = False
        self.pre_logits = nn.Identity()

        # Classifier head(s)就是全连接层
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Weight init权重初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        x=self.R50(x)
        # [B, C, H, W] -> [B, num_patches, embed_dim]这里就是patch_embedding层
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]拼接，成了197*768

        x = self.pos_drop(x + self.pos_embed)#相加操作

        x = self.blocks(x)#通过一系列Encoder Block层
        x = self.norm(x)#通过一个norm层

        return self.pre_logits(x[:, 0])


    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)#这里是直接到这里了
        return x



def vit_base_patch32_224(num_classes: int = 1000):


    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              num_classes=num_classes)
    return model