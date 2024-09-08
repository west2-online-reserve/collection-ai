import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel = 1, stride = 1, padding = 0, act = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.SiLU() if act else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, shortcut = True, e = 0.5) -> None:
        super().__init__()
        hidden_channel = int(out_channel * e)
        self.conv1 = Conv(in_channel, hidden_channel)
        self.conv2 = Conv(hidden_channel, out_channel, 3, 1, 1)
        self.add = shortcut and in_channel == out_channel
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class C3(nn.Module):
    def __init__(self, in_channel, out_channel, num = 1, shortcut = True, e = 0.5) -> None:
        super().__init__()
        hidden_channel = int(out_channel * e)
        self.conv1 = Conv(in_channel, hidden_channel)
        self.conv2 = Conv(in_channel, hidden_channel)
        self.conv3 = Conv(hidden_channel * 2, out_channel)
        self.m = nn.Sequential(*(Bottleneck(hidden_channel, hidden_channel, shortcut, e=1) for _ in range(num)))
    
    def forward(self, x):
        return self.conv3(torch.cat((self.conv1(x), self.m(self.conv2(x))), dim=1))

class SPPF(nn.Module):
    def __init__(self, in_channel, out_channel, k = 5) -> None:
        super().__init__()
        hidden_channel = in_channel // 2
        self.conv1 = Conv(in_channel, hidden_channel)
        self.conv2 = Conv(hidden_channel * 4, out_channel)
        self.m = nn.MaxPool2d(kernel_size = k, stride = 1, padding = k // 2)
    
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.conv2(torch.cat((x, y1, y2, self.m(y2)), dim = 1))
