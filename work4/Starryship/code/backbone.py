import torch
import torch.nn as nn
import torch.nn.functional as F

class_all = (
    "SiLU",
    "DropPath",
    "Conv",
    "DFL",
    "SwinTransformerBlock",
    "WindowAttention",
    "Mlp",
    "SwinTransformerLayer",
    "Bottleneck",
    "C2f",
    "C3",
    "C3STR",
    "SPPF",
    "Backbone",
)

def_all=(
    "autopad",
)


def autopad(k, p=None, d=1):  
    # kernel, padding, dilation
    # 对输入的特征层进行自动padding，按照Same原则
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class SiLU(nn.Module):  
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
        
class Conv(nn.Module):
    """
    标准卷积块，包括卷积、标准化和激活函数。

    Args:
        c1 (int): 输入通道数。
        c2 (int): 输出通道数。
        k (int or tuple, optional): 卷积核大小。默认为1。
        s (int, optional): 卷积步长。默认为1。
        p (int or tuple, optional): 补充大小。默认为None。
        g (int, optional): 分组卷积数。默认为1。
        d (int, optional): 卷积核元素之间的间距。默认为1。
        act (bool or nn.Module, optional): 是否使用激活函数。默认为True。

    Attributes:
        conv (nn.Conv2d): 卷积层。
        bn (nn.BatchNorm2d): 标准化层。
        act (nn.Module): 激活函数。

    Examples:
        >>> conv = Conv(3, 64, 3, 1, 1)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> output = conv(x)
    """
    default_act = SiLU() 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class DFL(nn.Module):
    """
    DFL模块，用于Distribution Focal Loss（DFL）。

    Args:
        c1 (int, optional): 输入通道数。默认为16。

    Attributes:
        conv (nn.Conv2d): 用于计算DFL的卷积层。
        c1 (int): 输入通道数。

    Examples:
        >>> dfl = DFL(c1=16)
        >>> x = torch.randn(1, 16, 8400)
        >>> output = dfl(x)
    """
    def __init__(self, c1=16):
        super().__init__()
        self.conv   = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x           = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1     = c1

    def forward(self, x):
        # bs, self.reg_max * 4, 8400
        b, c, a = x.shape
        # bs, 4, self.reg_max, 8400 => bs, self.reg_max, 4, 8400 => b, 4, 8400
        # 以softmax的方式，对0~16的数字计算百分比，获得最终数字。
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

# class Attention(nn.Module):
#     #添加多头注意力机制
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads  = num_heads
#         self.scale      = (dim // num_heads) ** -0.5

#         self.qkv        = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop  = nn.Dropout(attn_drop)
#         self.proj       = nn.Linear(dim, dim)
#         self.proj_drop  = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, W,H, C     = x.shape
#         qkv         = self.qkv(x).reshape(B,3, self.num_heads, W*H*C // self.num_heads).permute(1, 0, 2, 3)
#         q, k, v     = qkv[0], qkv[1], qkv[2]

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, W,H, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
 
class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer模块，包含卷积层和一系列Swin Transformer层。

    Args:
        c1 (int): 输入通道数。
        c2 (int): 输出通道数。
        num_heads (int): 注意力头的数量。
        num_layers (int): Swin Transformer层的数量。
        window_size (int, optional): 窗口大小，默认为8。
    """
    def __init__(self, c1, c2, num_heads, num_layers, window_size=8):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
 
        # remove input_resolution
        self.blocks = nn.Sequential(*[SwinTransformerLayer(dim=c2, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2) for i in range(num_layers)])
 
    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.blocks(x)
        return x
        
class WindowAttention(nn.Module):
    """
    基于窗口的注意力机制模块。

    Args:
        dim (int): 输入和输出特征的维度。
        window_size (tuple): 窗口的大小，格式为(Wh, Ww)。
        num_heads (int): 注意力头的数量。
        qkv_bias (bool, optional): 是否使用偏置。默认为True。
        qk_scale (float, optional): QK缩放因子。默认为None。
        attn_drop (float, optional): 注意力机制的dropout率。默认为0.。
        proj_drop (float, optional): 用于投影的dropout率。默认为0.。

    Attributes:
        dim (int): 输入和输出特征的维度。
        window_size (tuple): 窗口的大小，格式为(Wh, Ww)。
        num_heads (int): 注意力头的数量。
        scale (float): 缩放因子。
        relative_position_bias_table (torch.Parameter): 相对位置偏置表。
        relative_position_index (torch.Tensor): 相对位置索引。
        qkv (torch.nn.Linear): 用于计算Q、K、V的线性层。
        attn_drop (torch.nn.Dropout): 注意力机制的dropout层。
        proj (torch.nn.Linear): 投影线性层。
        proj_drop (torch.nn.Dropout): 用于投影的dropout层。
        softmax (torch.nn.Softmax): Softmax层用于计算注意力权重。
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
 
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
 
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
 
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
 
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
 
        nn.init.normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
 
    def forward(self, x, mask=None):
 
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
 
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
 
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
 
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
 
        attn = self.attn_drop(attn)
        
        try:
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        except:
            #print(attn.dtype, v.dtype)
            x = (attn.half() @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
 
class Mlp(nn.Module):
    """
    多层感知机（MLP）模块，包含两个线性层和激活函数。

    Args:
        in_features (int): 输入特征的维度。
        hidden_features (int, optional): 隐藏层特征的维度，默认为None，表示与输入特征维度相同。
        out_features (int, optional): 输出特征的维度，默认为None，表示与输入特征维度相同。
        act_layer (nn.Module, optional): 激活函数，默认为nn.SiLU。
        drop (float, optional): Dropout概率，默认为0.0。
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
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

class SwinTransformerLayer(nn.Module):
    """
    Swin Transformer中的一个层。

    Args:
        dim (int): 特征维度。
        num_heads (int): 注意力头的数量。
        window_size (int, optional): 窗口大小。默认为8。
        shift_size (int, optional): 移动大小。默认为0。
        mlp_ratio (float, optional): MLP扩展比率。默认为4.。
        qkv_bias (bool, optional): 是否使用注意力偏置。默认为True。
        qk_scale (float, optional): QK缩放因子。默认为None。
        drop (float, optional): 用于全连接层和注意力层的dropout率。默认为0.。
        attn_drop (float, optional): 注意力层的dropout率。默认为0.。
        drop_path (float, optional): DropPath的概率。默认为0.。
        act_layer (torch.nn.Module, optional): 激活函数层。默认为nn.SiLU。
        norm_layer (torch.nn.Module, optional): 归一化层。默认为nn.LayerNorm。

    Attributes:
        dim (int): 特征维度。
        num_heads (int): 注意力头的数量。
        window_size (int): 窗口大小。
        shift_size (int): 移动大小。
        mlp_ratio (float): MLP扩展比率。
        norm1 (torch.nn.Module): 第一个归一化层。
        attn (torch.nn.Module): 注意力层。
        drop_path (torch.nn.Module): DropPath层。
        norm2 (torch.nn.Module): 第二个归一化层。
        mlp (torch.nn.Module): MLP层。
    """
    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.SiLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
 
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
 
    def create_mask(self, H, W):
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
 
        def window_partition(x, window_size):

            B, H, W, C = x.shape
            x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
            windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
            return windows
 
        def window_reverse(windows, window_size, H, W):
            
            B = int(windows.shape[0] / (H * W / window_size / window_size))
            x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
            return x
 
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
 
        return attn_mask
 
    def forward(self, x):
        # reshape x[b c h w] to x[b l c]
        _, _, H_, W_ = x.shape
 
        Padding = False
        if min(H_, W_) < self.window_size or H_ % self.window_size!=0 or W_ % self.window_size!=0:
            Padding = True
            # print(f'img_size {min(H_, W_)} is less than (or not divided by) window_size {self.window_size}, Padding.')
            pad_r = (self.window_size - W_ % self.window_size) % self.window_size
            pad_b = (self.window_size - H_ % self.window_size) % self.window_size
            x = F.pad(x, (0, pad_r, 0, pad_b))
 
        # print('2', x.shape)
        B, C, H, W = x.shape
        L = H * W
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)  # b, L, c
 
        # create mask from init to forward
        if self.shift_size > 0:
            attn_mask = self.create_mask(H, W).to(x.device)
        else:
            attn_mask = None
 
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
 
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
 
        def window_partition(x, window_size):

            B, H, W, C = x.shape
            x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
            windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
            return windows
 
        def window_reverse(windows, window_size, H, W):

            B = int(windows.shape[0] / (H * W / window_size / window_size))
            x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
            return x
 
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
 
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C
 
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
 
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
 
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
 
        x = x.permute(0, 2, 1).contiguous().view(-1, C, H, W)  # b c h w
 
        if Padding:
            x = x[:, :, :H_, :W_]  # reverse padding
 
        return x

class Bottleneck(nn.Module):
    """
    C2f和C3中的瓶颈块。

    Args:
        c1 (int): 输入通道数。
        c2 (int): 输出通道数。
        shortcut (bool, optional): 是否使用快捷连接。默认为True。
        g (int, optional): 分组卷积的组数。默认为1。
        k (tuple, optional): 卷积核大小。默认为(3, 3)。
        e (float, optional): 瓶颈块的扩展因子。默认为0.5。

    Attributes:
        cv1 (torch.nn.Module): 第一个卷积层。
        cv2 (torch.nn.Module): 第二个卷积层。
        add (bool): 是否应用快捷连接。
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))    

class C2f(nn.Module):
    """
    C2f模块,CSPNet结构结构，大残差结构

    Args:
        c1 (int): 输入通道数。
        c2 (int): 输出通道数。
        n (int, optional): Bottleneck块的数量。默认为1。
        shortcut (bool, optional): 是否使用快捷连接。默认为False。
        g (int, optional): 分组卷积的组数。默认为1。
        e (float, optional): Bottleneck块的扩展因子。默认为0.5。

    Attributes:
        c (int): 隐藏通道数。
        cv1 (torch.nn.Module): 第一个卷积层。
        cv2 (torch.nn.Module): 第二个卷积层。
        m (torch.nn.ModuleList): Bottleneck块的列表。
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c      = int(c2 * e) 
        self.cv1    = Conv(c1, 2 * self.c, 1, 1)
        self.cv2    = Conv((2 + n) * self.c, c2, 1)
        self.m      = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # 进行一个卷积，然后划分成两份，每个通道都为c
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3(nn.Module):
    """
    C3模块。

    Args:
        c1 (int): 输入通道数。
        c2 (int): 输出通道数。
        n (int, optional): Bottleneck块的数量。默认为1。
        shortcut (bool, optional): 是否使用快捷连接。默认为True。
        g (int, optional): 分组卷积的组数。默认为1。
        e (float, optional): Bottleneck块的扩展因子。默认为0.5。

    Attributes:
        cv1 (torch.nn.Module): 第一个卷积层。
        cv2 (torch.nn.Module): 第二个卷积层。
        cv3 (torch.nn.Module): 第三个卷积层。
        m (torch.nn.Sequential): Bottleneck块的序列容器。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):

        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):

        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3STR(C3):
    """
    C3STR模块，基于Swin Transformer块的C3模块。

    Args:
        c1 (int): 输入通道数。
        c2 (int): 输出通道数。
        n (int, optional): Bottleneck块的数量。默认为1。
        shortcut (bool, optional): 是否使用快捷连接。默认为True。
        g (int, optional): 分组卷积的组数。默认为1。
        e (float, optional): Bottleneck块的扩展因子。默认为0.5。

    Attributes:
        m (SwinTransformerBlock): SwinTransformerBlock对象。
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)


class SPPF(nn.Module):
    """
    SPPF模块，实现空间金字塔池化（SPP）结构，5、9、13最大池化核的最大池化。

    Args:
        c1 (int): 输入通道数。
        c2 (int): 输出通道数。
        k (int, optional): 最大池化核的大小。默认为5。

    Attributes:
        cv1 (Conv): 输入通道数到c_的1x1卷积。
        cv2 (Conv): c_ * 4到输出通道数的1x1卷积。
        m (MaxPool2d): 最大池化操作，使用指定的池化核大小。
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_          = c1 // 2
        self.cv1    = Conv(c1, c_, 1, 1)
        self.cv2    = Conv(c_ * 4, c2, 1, 1)
        self.m      = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Backbone(nn.Module):
    """
    Backbone骨干网络

    Args:
        base_channels (int): 基础通道数。
        base_depth (int): 基础深度。
        deep_mul (int): 深度乘数。
        phi (int): 基于网络的尺寸。
        pretrained (bool, optional): 是否使用预训练模型。默认为False。

    Attributes:
        stem (Conv): 输入层到第一个特征层的卷积块。
        dark2 (Sequential): 第二个特征层的卷积块。
        dark3 (Sequential): 第三个特征层的卷积块。
        dark4 (Sequential): 第四个特征层的卷积块。
        dark5 (Sequential): 第五个特征层的卷积块。
    """
    def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
        super().__init__()
        #-----------------------------------------------#
        #   输入图片是3, 640, 640
        #-----------------------------------------------#
        # 3, 640, 640 => 32, 640, 640 => 64, 320, 320
        self.stem = Conv(3, base_channels, 3, 2)
        
        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C2f(base_channels * 2, base_channels * 2, base_depth, True),
        )
        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )
        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C3STR(base_channels * 8, base_channels * 8, base_depth * 2, True),
        )
        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            C3STR(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        #-----------------------------------------------#
        #   dark3的输出为256, 80, 80，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        #-----------------------------------------------#
        #   dark4的输出为512, 40, 40，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        #-----------------------------------------------#
        #   dark5的输出为1024 * deep_mul, 20, 20，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3

