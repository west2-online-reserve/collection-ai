import functools
import torch
import matplotlib.pyplot as plt
import torchvision.models
from torch import nn
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
import pandas as pd
from PIL import Image


# PatchEmbed层
class PatchEmbed(nn.Module):
    # 2D Image to Patch Embedding
    # norm_layer参数指定归一化层的类型
    def __init__(self, img_size=224, patch_size=16, in_channel=3, embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()
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
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

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
        out_features = out_features or in_features
        hidden_features = in_features or hidden_features
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
                 drop_ratio=0., attn_drop_ratio=0, drop=0., embed_layer=PatchEmbed,
                 norm_layer=None, act_layer=None):
        super(Vit, self).__init__()
        self.num_classes = num_classes
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
        # nn.init.xavier_normal(self.pos_embed)
        # nn.init.xavier_normal(self.cls_token)
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


# 导入数据集
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((230, 230)),
        transforms.RandomRotation(15, ),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]),
}

categories = {
    'accordion': 0, 'airplanes': 1, 'anchor': 2, 'ant': 3, 'yin_yang': 4, 'barrel': 5,
    'bass': 6, 'beaver': 7, 'binocular': 8, 'bonsai': 9, 'brain': 10, 'brontosaurus': 11,
    'buddha': 12, 'butterfly': 13, 'camera': 14, 'cannon': 15, 'car_side': 16, 'ceiling_fan': 17,
    'cellphone': 18, 'chair': 19, 'chandelier': 20, 'cougar_body': 21, 'cougar_face': 22,
    'crab': 23, 'crayfish': 24, 'crocodile': 25, 'crocodile_head': 26, 'cup': 27, 'dalmatian': 28,
    'dollar_bill': 29, 'dolphin': 30, 'dragonfly': 31, 'electric_guitar': 32, 'elephant': 33,
    'emu': 34, 'euphonium': 35, 'ewer': 36, 'Faces': 37, 'Faces_easy': 38, 'ferry': 39, 'flamingo': 40,
    'flamingo_head': 41, 'garfield': 42, 'gerenuk': 43, 'gramophone': 44, 'grand_piano': 45,
    'hawksbill': 46, 'headphone': 47, 'hedgehog': 48, 'helicopter': 49, 'ibis': 50, 'inline_skate': 51,
    'joshua_tree': 52, 'kangaroo': 53, 'ketch': 54, 'lamp': 55, 'laptop': 56, 'Leopards': 57, 'wrench': 58,
    'llama': 59, 'lobster': 60, 'lotus': 61, 'mandolin': 62, 'mayfly': 63, 'menorah': 64, 'metronome': 65,
    'minaret': 66, 'Motorbikes': 67, 'nautilus': 68, 'octopus': 69, 'okapi': 70, 'pagoda': 71, 'panda': 72,
    'pigeon': 73, 'pizza': 74, 'platypus': 75, 'pyramid': 76, 'revolver': 77, 'rhino': 78, 'rooster': 79,
    'saxophone': 80, 'schooner': 81, 'scissors': 82, 'scorpion': 83, 'sea_horse': 84, 'snoopy': 85,
    'soccer_ball': 86, 'stapler': 87, 'starfish': 88, 'stegosaurus': 89, 'stop_sign': 90, 'strawberry': 91,
    'sunflower': 92, 'tick': 93, 'trilobite': 94, 'umbrella': 95, 'watch': 96, 'water_lilly': 97, 'wheelchair': 98,
    'wild_cat': 99, 'windsor_chair': 100
}


# 自定义数据集
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform
        self.image = []
        for idx in range(len(self.img_labels)):
            img = Image.open(self.img_labels.iloc[idx, 0])
            # 若为单通道图像，则复制为三通道图像
            if len(img.split()) == 1:
                r, g, b = img, img, img
                img = Image.merge("RGB", (r, g, b))
            tmp = self.transform(img)
            self.image.append(tmp)
        #  = [self.transform(Image.open(self.img_labels.iloc[idx,0]))
        #               for idx in range(len(self.img_labels))]
        for i in range(len(self.img_labels)):
            self.img_labels.iloc[i, 1] = categories[self.img_labels.iloc[i, 1]]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.image[idx]
        label = self.img_labels.iloc[idx, 1]
        # print(label)
        return image, label


# 路径地址
# data_dir = '/kaggle/output/caltech101'
# img_dir = ".\caltech_images"
train_dataset = CustomImageDataset(annotations_file="./train_annotations.csv",
                                   transform=data_transforms['train'])
test_dataset = CustomImageDataset(annotations_file="./test_annotations.csv",
                                  transform=data_transforms['test'])


def load_data_caltech101(batch_size):
    return (data.DataLoader(train_dataset, batch_size, shuffle=True),
            data.DataLoader(test_dataset, batch_size, shuffle=False))


def train(net, epoch_num, loss, updater, trainloader, testloader, device):
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    for epoch in range(epoch_num):
        print("-----第{}轮训练开始------".format(epoch + 1))
        train_loss = 0.0
        test_loss = 0.0
        train_sum, train_cor, test_sum, test_cor = 0.0, 0.0, 0.0, 0.0
        # 开始训练
        if isinstance(net, nn.Module):
            net.train()
        for i, data in enumerate(trainloader):
            X, label = data[0].to(device), data[1].to(device)
            updater.zero_grad()
            y_hat = net(X)
            l1 = loss(y_hat, label)
            l1.mean().backward()
            updater.step()
            # 计算每轮训练集的loss
            train_loss += l1.item()
            # 计算训练集精度
            _, predicted = torch.max(y_hat.data, 1)
            train_cor += (predicted == label).sum().item()
            train_sum += label.size(0)

        # 进入测试模式
        if isinstance(net, nn.Module):
            net.eval()
        for j, data in enumerate(testloader):
            X, label = data[0].to(device), data[1].to(device)
            y_hat = net(X)
            l2 = loss(y_hat, label)
            test_loss += l2.item()
            _, predicted = torch.max(y_hat.data, 1)
            test_cor += (predicted == label).sum().item()
            test_sum += label.size(0)

        train_loss_list.append(train_loss / i)
        train_acc_list.append(train_cor / train_sum * 100)
        test_loss_list.append(test_loss / j)
        test_acc_list.append(test_cor / test_sum * 100)
        print("Train loss:{}   Train accuracy:{}%  Test loss:{}  Test accuracy:{}%".format(
            train_loss / i, train_cor / train_sum * 100, test_loss / j, test_cor / test_sum * 100))
    return train_loss_list, train_acc_list, test_loss_list, test_acc_list


def show(train_acc_list, test_acc_list, train_loss_list, test_loss_list):
    # 创建准确率画布
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1 = plt.plot(range(len(train_acc_list)), train_acc_list, 'red')
    line2 = plt.plot(range(len(test_acc_list)), test_acc_list, 'green')
    # 设置横纵坐标
    ax.set_xlabel('epochs', fontsize=14)
    ax.set_ylabel('accuracy rate(%)', fontsize=14)
    # 共用x轴
    ax2 = ax.twinx()
    line3 = ax2.plot(range(len(train_loss_list)), train_loss_list, 'blue')
    line4 = ax2.plot(range(len(test_loss_list)), test_loss_list, 'yellow')
    ax2.set_ylabel('loss value', fontsize=14)
    # 合并图例
    # ax.legend(lines, labs, loc=0)
    ax.legend(['train accuracy', 'test accuracy'], loc='best')
    ax2.legend(['train loss', 'test loss'], loc='best')
    # 生成网格线
    plt.grid()
    plt.savefig('Caltech101_fig_1')
    plt.show()


# 构建模型
def vit_base_patch16_224(num_classes: int = 101):
    model = Vit(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
                num_classes=num_classes, qkv_bias=True)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = vit_base_patch16_224(num_classes=101)
    net = torchvision.models.vit_b_16(pretrained=True)
    print(net)
    # net.load_state_dict(torch.load('Caltech101.pth'))
    net.to(device)
    loss = nn.CrossEntropyLoss()
    loss.to(device)
    updater = torch.optim.Adam(net.parameters(), lr=0.001)
    # 读取数据集
    batch_size = 64
    train_iter, test_iter = load_data_caltech101(batch_size)
    # 开始训练
    epoch_num = 5
    train_loss_list, train_acc_list, test_loss_list, test_acc_list = \
        train(net, epoch_num, loss, updater, train_iter, test_iter, device)
    torch.save(net.state_dict(), 'Caltech101.pth')
    show(train_acc_list, test_acc_list, train_loss_list, test_loss_list)

if __name__ == "__main__":
    main()
