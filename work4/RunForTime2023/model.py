import torch


class block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.1,inplace=True))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels))
        self.shortcut = torch.nn.Sequential()
        self.activation = torch.nn.LeakyReLU(0.1,inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                torch.nn.BatchNorm2d(out_channels))

    def forward(self, x):
        temp = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += self.shortcut(temp)
        x = self.activation(x)
        return x


class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.1,inplace=True),
            torch.nn.MaxPool2d(3, 2, 1))
        self.block2 = torch.nn.Sequential(
            block(64, 64, 1),
            block(64, 64, 1))
        self.block3 = torch.nn.Sequential(
            block(64, 128, 2),
            block(128, 128, 1))
        self.block4 = torch.nn.Sequential(
            block(128, 256, 2),
            block(256, 256, 1))
        self.block5 = torch.nn.Sequential(
            block(256, 512, 2),
            block(512, 512, 1))
        self.block6 = torch.nn.Sequential(
            torch.nn.AvgPool2d(2,2),
            torch.nn.Conv2d(512,30,1),
            torch.nn.BatchNorm2d(30),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x.permute(0,2,3,1)