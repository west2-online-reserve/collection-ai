import torch
import torch.nn as nn
from models.common import Conv, C3, SPPF

class backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n1 = nn.Sequential(
            Conv(3, 32, 6, 2, 2), # / 2 = 320
            Conv(32, 64, 3, 2, 1), # / 2 = 160
            C3(64, 64),
            Conv(64, 128, 3, 2, 1), # / 2 = 80
            C3(128, 128),
            C3(128, 128)
        )
        self.n2 = nn.Sequential(
            Conv(128, 256, 3, 2, 1), # / 2 = 40
            C3(256, 256),
            C3(256, 256),
            C3(256, 256)
        )
        self.n3 = nn.Sequential(
            Conv(256, 512, 3, 2, 1), # / 2 = 20
            C3(512, 512),
            SPPF(512, 512)
        )
    
    def forward(self, x):
        y1 = self.n1(x)
        y2 = self.n2(y1)
        return y1, y2, self.n3(y2)

class head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor = 2)
        self.conv1 = Conv(512, 256)
        self.conv2 = nn.Sequential(
            C3(512, 256),
            Conv(256, 128)
        )
        self.C3_1 = C3(256, 256)
        self.conv_1 = Conv(256, 128, 3, 2, 1)
        self.C3_2 = C3(256, 256)
        self.conv_2 = Conv(256, 256, 3, 2, 1)
        self.C3_3 = C3(512, 512)
        self.head1 = Conv(256, 75)
        self.head2 = Conv(256, 75)
        self.head3 = Conv(512, 75)
    
    def forward(self, x1, x2, x3):
        y1 = self.conv1(x3)
        y2 = self.conv2(torch.cat((x2, self.upsample(y1)), dim = 1))
        y3 = self.C3_1(torch.cat((x1, self.upsample(y2)), dim = 1))
        y4 = self.C3_2(torch.cat((y2, self.conv_1(y3)), dim = 1))
        y5 = self.C3_3(torch.cat((y1, self.conv_2(y4)), dim = 1))
        return self.head1(y3).view(y3.shape[0], 3, y3.shape[2], y3.shape[3], -1), self.head2(y4).view(y4.shape[0], 3, y4.shape[2], y4.shape[3], -1), self.head3(y5).view(y5.shape[0], 3, y5.shape[2], y5.shape[3], -1) # / 80, 40, 20

class YOLO(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = backbone()
        self.head = head()
    
    def forward(self, x):
        y1, y2, y3 = self.backbone(x)
        return self.head(y1, y2, y3)
