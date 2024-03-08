import functools
import torch
from torch import nn
from ..data.datasetsProcess import load_data_caltech101
from ..trainer.train import train
from ..utils.util import show

# PatchEmbed层
class patch_embed(nn.Module):
    # 2D Image to Patch Embedding
    # norm_layer参数指定归一化层的类型
    def __init__(self, img_size=224, patch_size=16, in_channel=3, embed_dim=768, norm_layer=None):
        super(patch_embed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        # 224整除16=14
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # 14*14=196
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # 定义卷积层,输出为768，卷积核为16×16，步长16
        self.proj = nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 归一化层
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # flatten: [B,C,H,W] -> [B,C,HW]
        # transpose: [B,C,HW] -> [B,HW,C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


# Attention(Q,K,V) = softmax(Q*K^T/dk^0.5)V
class MutiHeadsAttention(nn.Module):
    def __init__(self,
                 dim,  # token维度
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(MutiHeadsAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # 使用一个全连接层得到qkv,加速并行处理
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [ batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        # qkv(): ->[batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads，embed_dim_per_head]
        # permute: ->[ 3,batch_size，num_heads，num_patches + 1， embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads,num_patches + 1, embed_dim _per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose: ->[batch_size，num_heads，embed_dim_per_head，num_patches + 1]
        # @: multiply ->[batch_size，num_heads，num patches + 1，num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 对每一行进行softmax处理
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply ->[batch_size，num_heads，num_patches + 1,embed_dim_per_head]
        # transpose: ->[batch_size，num _patches + 1, num_heads，embed_dim per_head]
        # reshape: ->[ batch_size, num patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 全连接层
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
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
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_ratio1=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MutiHeadsAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.dropout = nn.Dropout(drop_ratio1)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        y = self.norm1(x)
        y = self.attn(y)
        x = x + self.dropout(y)
        y = self.norm2(x)
        y = self.mlp(y)
        x = x + self.dropout(y)
        return x


class Vit(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channel=3, num_classes=101,
                 embed_dim=768, depth=12,  # depth为block堆叠次数
                 num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 # representation_size=0.,
                 drop_ratio=0., attn_drop_ratio=0, drop=0., embed_layer=patch_embed,
                 norm_layer=None, act_layer=None):
        super(Vit, self).__init__()
        self.num_classes = num_classes
        # 卷积核个数，隐藏层 768
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        # 使用partial进行参数绑定，返回参数缩减版本
        norm_layer = norm_layer or functools.partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        # embed层
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_channel=in_channel,
                                       embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # 定义class_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        # 等差序列：分割点数为depth，开始值为0，结束值为drop
        dpr = [x.item() for x in torch.linspace(0, drop, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_ratio1=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(self.num_features, num_classes)
        # 权重初始化(截断正态分布:tensor, mean=0.0, std=1.0, a=- 2.0, b=2.0)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # 调用vit初始函数
        # pytorch中的model.apply(fn)
        # 会递归地将函数fn应用到父模块的每个子模块submodule，也包括model这个父模块自身。经常用于初始化init_weights的操作。
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B,C,H,w]->[B,num patches, embed_dim]
        x = self.patch_embed(x)  # [B,196,768]
        # [1,1,768] -> [B,1,768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # 进行拼接
        x = torch.cat((cls_token, x), dim=1)  # [B,197,768]
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


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

# 构建模型
def vit_base_patch16_224(num_classes: int = 101):
    model = Vit(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
                num_classes=num_classes, qkv_bias=True)
    return model

def vit_large_patch16_224(num_classes: int = 101):
    model = Vit(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16,
                num_classes=num_classes, qkv_bias=True)
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = vit_base_patch16_224(num_classes=101)
    # net = vit_large_patch16_224(num_classes=101)
    # net = torchvision.models.vit_b_16(pretrained=True)
    net.load_state_dict(torch.load('Caltech101_1.pth'))
    net.to(device)
    loss = nn.CrossEntropyLoss()
    loss.to(device)

    # 读取数据集
    batch_size = 64
    train_iter, test_iter = load_data_caltech101(batch_size)

    # 开始训练
    epoch_num = 20
    # 相较于Adam，收敛速度更快，正则系数不受动量影响
    updater = torch.optim.AdamW(net.parameters(), lr=0.1e-4, weight_decay=0.05)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(updater, T_max=epoch_num * batch_size)
    # updater = torch.optim.SGD(net.parameters(), lr=0.01)
    train_loss_list, train_acc_list, test_loss_list, test_acc_list = \
        train(net, epoch_num, loss, updater, train_iter, test_iter, device)
    torch.save(net.state_dict(), 'Caltech101.pth')
    show(train_acc_list, test_acc_list, train_loss_list, test_loss_list)

if __name__ == "__main__":
    main()
