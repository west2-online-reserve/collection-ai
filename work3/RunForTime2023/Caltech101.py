import torch
import torchvision
from matplotlib import pyplot
import os

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            torch.nn.BatchNorm2d(out_channels))
        self.shortcut = torch.nn.Sequential()
        self.relu = torch.nn.ReLU()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                torch.nn.BatchNorm2d(out_channels))
    def forward(self, x):
        temp = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += self.shortcut(temp)
        x = self.relu(x)
        return x

class Caltech(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
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
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(512, 101))
        # self.upsample1 = torch.nn.ConvTranspose2d(101, 101, 2, 2)
        # self.upsample2 = torch.nn.ConvTranspose2d(101, 101, 16, 16)
        self.upsample1 = torch.nn.ConvTranspose2d(101, 101, 4, 2, 1)
        self.upsample2 = torch.nn.ConvTranspose2d(101, 101, 32, 16, 8)
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(256, 101, 1), torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(512, 101, 1), torch.nn.ReLU())
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        temp1 = x
        x = self.block5(x)
        temp2 = x
        temp1 = self.conv1(temp1)
        temp2 = self.conv2(temp2)
        output1 = temp1 + self.upsample1(temp2)
        output1 = self.upsample2(output1)
        x = self.block6(x)
        output2 = torch.nn.functional.log_softmax(x, dim=1)
        return output1, output2

batchSize = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learningRate = 0.01
turns = 10
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.RandomHorizontalFlip(0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# 由于下载地址迁移，直接引用Caletch101类下载失败，故只能从本地导入数据。导入前须删除BACKGROUND_Google文件夹
dataSet = torchvision.datasets.ImageFolder("Caltech101/caltech101/101_ObjectCategories", transform=transform)
dataSetLoader = torch.utils.data.DataLoader(dataSet, batch_size=batchSize, shuffle=True, num_workers=4, pin_memory=True)
model = Caltech().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
trainLoss = [0, ] * turns
trainAccuracy = [0, ] * turns
labels = ["Faces", "Faces_easy", "Leopards", "Motorbikes", "accordion", "airplanes", "anchor", "ant", "barrel", "bass",
          "beaver", "binocular", "bonsai", "brain", "brontosaurus", "buddha", "butterfly", "camera", "cannon",
          "car_side", "ceiling_fan", "cellphone", "chair", "chandelier", "cougar_body", "cougar_face", "crab",
          "crayfish", "crocodile", "crocodile_head", "cup", "dalmatian", "dollar_bill", "dolphin", "dragonfly",
          "electric_guitar", "elephant", "emu", "euphonium", "ewer", "ferry", "flamingo", "flamingo_head", "garfield",
          "gerenuk", "gramophone", "grand_piano", "hawksbill", "headphone", "hedgehog", "helicopter", "ibis",
          "inline_skate", "joshua_tree", "kangaroo", "ketch", "lamp", "laptop", "llama", "lobster", "lotus", "mandolin",
          "mayfly", "menorah", "metronome", "minaret", "nautilus", "octopus", "okapi", "pagoda", "panda", "pigeon",
          "pizza", "platypus", "pyramid", "revolver", "rhino", "rooster", "saxophone", "schooner", "scissors",
          "scorpion", "sea_horse", "snoopy", "soccer_ball", "stapler", "starfish", "stegosaurus", "stop_sign",
          "strawberry", "sunflower", "tick", "trilobite", "umbrella", "watch", "water_lilly", "wheelchair", "wild_cat",
          "windsor_chair", "wrench", "yin_yang"]
counts = [435, 435, 200, 798, 55, 800, 42, 42, 47, 54, 46, 33, 128, 98, 43, 85, 91, 50, 43, 123, 47, 59, 62, 107, 47,
          69, 73, 70, 50, 51, 57, 67, 52, 65, 68, 75, 64, 53, 64, 85, 67, 67, 45, 34, 34, 51, 99, 100, 42, 54, 88, 80,
          31, 64, 86, 114, 61, 81, 78, 41, 66, 43, 40, 87, 32, 76, 55, 35, 39, 47, 38, 45, 53, 34, 57, 82, 59, 49, 40,
          63, 39, 84, 57, 35, 64, 45, 86, 59, 64, 35, 85, 49, 86, 75, 239, 37, 59, 34, 56, 39, 60]

model.train()
for i in range(turns):
    loss2 = 0.0
    for data, label in dataSetLoader:
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output1, output2 = model(data)
        loss = torch.nn.CrossEntropyLoss()
        loss = loss(output2, label)
        loss2 += loss
        loss.backward()
        optimizer.step()

if not os.path.exists("heatmap"):  # 保存图片时路径上的文件夹必须存在
    os.mkdir("heatmap")
model.eval()
dataSetLoader = torch.utils.data.DataLoader(dataSet, batch_size=batchSize, shuffle=False) # 顺序加载，方便判断文件名
count = 0
with torch.no_grad():
    for data, label in dataSetLoader:
        data = data.to(device)
        label = label.to(device)
        output1, output2 = model(data)
        for i in range(data.shape[0]):
            temp = torch.sum(output1[i], 0)
            temp = (temp - torch.max(temp)) / (torch.max(temp) - torch.min(temp)) * 255  # 转为RGB
            temp = temp.cpu().numpy().astype("uint8")
            count += 1
            if not os.path.exists("heatmap/" + labels[label[i]]):
                os.mkdir("heatmap/" + labels[label[i]])
            pyplot.figure(figsize=(56,56))
            pyplot.imshow(temp, cmap='coolwarm')
            pyplot.axis('off')
            pyplot.savefig("heatmap/" + labels[label[i]] + "/image_" + str(count).zfill(4) + ".jpg", bbox_inches='tight', pad_inches=0)
            pyplot.close()
            if count == counts[label[i]]:
                count = 0