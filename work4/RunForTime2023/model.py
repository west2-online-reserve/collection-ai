import torch


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU())
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels))
        self.shortcut = torch.nn.Sequential()
        self.activation = torch.nn.LeakyReLU()
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
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(3, 2, 1))
        self.block2 = torch.nn.Sequential(
            ResidualBlock(64, 64, 1),
            ResidualBlock(64, 64, 1))
        self.block3 = torch.nn.Sequential(
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 128, 1))
        self.block4 = torch.nn.Sequential(
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 256, 1))
        self.block5 = torch.nn.Sequential(
            ResidualBlock(256, 512, 2),
            ResidualBlock(512, 512, 1))
        self.block6 = torch.nn.Sequential(
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Conv2d(512, 30, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(30),
            torch.nn.Sigmoid()
        )
        # self.block6 = torch.nn.Sequential(
        #     torch.nn.AdaptiveAvgPool2d((1, 1)),
        #     torch.nn.Flatten(),
        #     torch.nn.Linear(512, 30))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = x.permute(0, 2, 3, 1)
        # output2 = torch.nn.functional.log_softmax(x, dim=1)
        return x
