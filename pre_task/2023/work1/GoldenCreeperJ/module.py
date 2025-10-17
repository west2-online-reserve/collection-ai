import torch


class my_module(torch.nn.Module):
    def __init__(self):
        super(my_module, self).__init__()
        self.module = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 3, 1, 1),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 128),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.module(x)
