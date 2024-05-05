import torch
from utils.dataloader import yoloDataset, yoloCollate
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from loss import Computeloss
from utils.mAP import ComputemAP
import torch.optim as optim
from models.yolo import YOLO
from trainer import train, test

batch_size = 16
lr = 2e-3
epoch = 100
anchors = [[[10, 13], [16, 30], [33, 23]],
           [[30, 61], [62, 45], [59, 119]],
           [[116, 90], [156, 198], [373, 326]]]
anchors[0] = [[j / 16 for j in i] for i in anchors[0]]
anchors[1] = [[j / 32 for j in i] for i in anchors[1]]
anchors[2] = [[j / 64 for j in i] for i in anchors[2]]
# print(anchors)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# device = "cpu"

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([320, 320])
])

train_dataset = yoloDataset("./data/VOCdevkit/VOC2012", True, transform = trans)
test_dataset = yoloDataset("./data/VOCdevkit/VOC2012", False, transform = trans)

train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = yoloCollate)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, collate_fn = yoloCollate)

model = YOLO().to(device)

crit = Computeloss(anchors, 20, device = device)
optims = optim.SGD(model.parameters(), lr)
cmAP = ComputemAP(anchors, device)

for i in range(epoch):
    train(model, i, train_dataloader, crit, optims, device, epoch)
    test(model, test_dataloader, crit, device, cmAP)
    if (i + 1) % 10 == 0:
        torch.save(model.state_dict(), "./m/" + str(i + 1) + ".pt")