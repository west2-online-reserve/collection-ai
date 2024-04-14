import math
from copy import copy
from pathlib import Path

import numpy as np

import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

# from utils.datasets import letterbox
# from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh
# from utils.plots import color_list, plot_one_box
# from utils.torch_utils import time_synchronized
import argparse
import logging
import sys
from copy import deepcopy
from utils1 import non_max_suppression

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)
try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None
# from models.common import *
# from models.experimental import *
# from utils.autoanchor import check_anchor_order
# from utils.general import make_divisible, check_file, set_logging
from utils import make_divisible



def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act) #math.gcd()返回的是最大公约数,这里就是group=1


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)#这里就调用了auropad
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())# act为true,就使用激活函数
        # 这里的act有的版本是silu有的版本是hardswish
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2 # shortcut就是残差连接，shortcut这个变量是在不同的Bottleneck有不同的处理方式

    def forward(self, x):
        # 根据是否shortcut为true输出不同的return
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


# 空间金字塔池化
class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    # k是一个元组
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        #对输入进行三次最大池化

    def forward(self, x):
        x = self.cv1(x)
        # 这里是3+1=4，就是一个没有操作的东西依次和不同最大池化的x进行拼接操作,注意
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

# focus和spp有点像，但是用的是slice切分操作
class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # 这个切片操作就是把最后两个维度2个采样1个，后面的两维度要除2
        # 给个例子，就是28*28变成了14*14
        # return self.conv(self.contract(x))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension # 沿着哪个维度进行拼接

    def forward(self, x):
        return torch.cat(x, self.d)


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    #anchors是[[10,13,16,30,33,23],[30,61,62,45,59,119],[116,90,156,198,373,326]]，
    # 每个anchor的都有两个值,分别代表长和宽,所以一共有几个锚框
    #ch是[128,256,512]

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor每个anchor输出的数目,就是80个类加上四个坐标信息和一个目标得分
        self.nl = len(anchors)  # number of detection layers就是每个特征图相应的层数，这里是3，因为他是个嵌套列表
        self.na = len(anchors[0]) // 2  # number of anchors每个层数的anchors数
        self.grid = [torch.zeros(1)] * self.nl  # init grid 初始化
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)#把anchor分开,[[[10,13],[16,30],[33,23]],[[.......]],[[.......]]]
        # 模型中需要保存下来的参数包括两种:
        # 一种是反向传播需要被optimizer更新的，称之为parameter
        # 一种是反向传播不需要更新的参数,称之为buffer
        # 不需要更新的参数,需要创建tensor，然后将tensor通过register_buffer()进行注册
        # 可以通过model.buffers()返回，注册完参数会自动保存到OrderDict中
        # buffer的更新在forward中,opti.step只能更新nn.parameter中去
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        # 这一步是构造1*1卷积，这里的x是构成通道的取值，这里是128,256,512
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        # 对nl进行迭代
        for i in range(self.nl):
            # 卷积运算
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # 调整顺序，变成连续的变量，把no（长度85，80个类加上四个坐标信息和一个目标得分）放到最后
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                # 如果不是在训练，就要用make_grid构造网格
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                #求出预测框的x，y，w，h信息
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

#锚框在三个chanel有不同的大小
#  - [10,13, 16,30, 33,23]  # P3/8，在8倍下采样的anchors大小比如第一个宽度是10,高度是13
#  - [30,61, 62,45, 59,119]  # P4/16，在16倍下采样的anchors大小
#  - [116,90, 156,198, 373,326]  # P5/32，在32倍下采样的anchors大小
class Model(nn.Module):
    def __init__(self, cfg='/tmp/pycharm_project_508/models/model1/yolov5s1.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        # 第一部分加载配置文件
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                # load完了这个self.yaml就是一个字典格式的
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model 第二部分利用配置文件搭建网络每一层
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels，在yaml文件里没有ch,就要这样给ch赋值，就是上面的3

        # 通过parse_model解析yaml文件构建模型
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        # 初始化names，给每一个类先给个名字，0,1,2,3,4,5....
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors 第三部分求网络的步长，求anchor的处理
        m = self.model[-1]  # Detect(),模型的最后一层
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            # 下采样的倍率[8,16,32]
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            # forward在这里直接新建了一张空白的图片，就
            # 是256*256的，3通道，通过一次前上传播，就知道stride的步长是8,16,32是这样得到stride的大小的

            # 把anchors的三个值分别除8,16,32,比如[10,13]除8-->[1.25,1.625]，anchor定义的是在原图上的大小，原图变小了，anchor也要变小，所以要除
            m.anchors /= m.stride.view(-1, 1, 1)
            # 检查anchor顺序是否与stride顺序一致
            self.stride = m.stride
            # self._initialize_biases()  # only run once初始化偏置
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases 第四部分 网络参数初始化并打印
        # initialize_weights(self)# 初始化权重
        # self.info() #  打印模型信息
        # logger.info('')

    def forward(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    #
    # def nms(self, mode=True):  # add or remove NMS module
    #     present = type(self.model[-1]) is NMS  # last layer is NMS
    #     if mode and not present:
    #         print('Adding NMS... ')
    #         m = NMS()  # module
    #         m.f = -1  # from
    #         m.i = self.model[-1].i + 1  # index
    #         self.model.add_module(name='%s' % m.i, module=m)  # add
    #         self.eval()
    #     elif not mode and present:
    #         print('Removing NMS... ')
    #         self.model = self.model[:-1]  # remove
    #     return self

#用来解析yaml文件 ，构建模型,d是字典
def parse_model(d, ch):  # model_dict, input_channels(3),这个d是yolov5.yaml,这个ch是一个列表[3]里面就一个3
    # logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))#打印yaml信息
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']#把这些东西全部取出来
    #因为anchors是有宽有高的，anchors6个值其实是三个高宽值，所以除二，na表示anchor数量，na是3
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors先看anchors是不是list，是list，一共有anchors[0],anchors[1],anchors[2]

    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)这个是模型输出的通道数，因为有三个anchor所以乘三，nc是有几个类，这里是80,5表示的是x,y,w,h,置信度

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out，layers等一下要用来储存层结构，save是统计哪些层是要保存的，因为到时候pan会有拼接操作，c2是输出的通道数
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args  开始遍历
        m = eval(m) if isinstance(m, str) else m  # eval strings其实这个‘Conv’，用eval函数推断这个东西，发现原来是common下的Conv类
        for j, a in enumerate(args):# 遍历args，j是枚举的次数，a是list
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings 同样的，判断list是不是str，不是就把list返回回去
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain注意，这里gd是深度倍数，当n(层的个数)，这个时候深度倍数就有用了gd=0.33，你的层数是3，其实在yolos就是1
        if m in [Conv, Bottleneck, DWConv, Focus,SPP,
                 C3]:# 现在看你的m到底是什么结构
            #如果m是Conv结构，就这样赋值
            c1, c2 = ch[f], args[0]#这个c1就是3，c2就是赋值的
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)#比如输入c2是64，还要乘一个通道倍数，其实是64*gw（0.5），最后是32，判断是不是8的倍数

            args = [c1, c2, *args[1:]] # 然后再把args写全，现在就有了c1，c2,[3,32,6,2,2],这下args全了
            if m in [ C3]:
                #注意，这里要加n进去，因为这里跟层数有关系
                args.insert(2, n)  # number of repeats假如是这两层的话，只传入一个参数,比如[128],所以要把c3额外加个n进去，就是在第二个位置，加个n进去
                n = 1
         # 如果x是Batchnorn，就这样赋值
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        # 下面都一样
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                # 把f中的输出累加到这层的channel
                args[1] = [list(range(args[1] * 2))] * len(f)
        # elif m is Contract:
        #     c2 = ch[f] * args[0] ** 2
        # elif m is Expand:
        #     c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module 这里要通过n来判断要多少个层
        t = str(m)[8:-2].replace('__main__.', '')  # module type判断模块里有没有那个字符串
        np = sum([x.numel() for x in m_.parameters()])  # number params统计每一层的参数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params给模型赋值
        # logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print打印输入信息
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist统计哪些层是需要保存的
        layers.append(m_)
        if i == 0:#对于第0层，需要把c2赋值进去，因为下一层需要用到这一层的输出通道作为输入通道
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save) # 返回赋值好的模型和需要保存的层数
