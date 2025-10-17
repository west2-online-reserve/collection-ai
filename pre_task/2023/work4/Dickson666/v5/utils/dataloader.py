import torch
from torch.utils.data import Dataset
import os
import xml.etree.ElementTree as ET
from PIL import Image

class_code = {"person": 0, "bird": 1, "cat": 2, "cow": 3, "dog": 4, "horse": 5, "sheep": 6, "aeroplane": 7,
              "bicycle": 8, "boat": 9, "bus": 10, "car": 11, "motorbike": 12, "train": 13, "bottle": 14,
              "chair": 15, "diningtable": 16, "pottedplant": 17, "sofa": 18, "tvmonitor": 19}

class yoloDataset(Dataset):
    def __init__(self, basedir, is_train = True, transform = None) -> None:
        super().__init__()
        listdir = os.path.join(basedir, "ImageSets", "Segmentation", "train.txt" if is_train else "val.txt")
        self.namelist = open(listdir).readlines()
        self.basedir = basedir
        self.transform = transform

    def __len__(self):
        return len(self.namelist)
    
    def __getitem__(self, index):
        namebase = self.namelist[index]
        xml_file = os.path.join(self.basedir, "Annotations", namebase[:-1] + ".xml")
        img_file = os.path.join(self.basedir, "JPEGImages", namebase[:-1] + ".jpg")
        tree = ET.parse(xml_file)
        W, H = float(tree.findtext("size/width")), float(tree.findtext("size/height"))
        cls, box = [], []
        for obj in tree.iter("object"):
            cls.append(class_code[obj.findtext("name")])
            xmin = float(obj.findtext("bndbox/xmin")) / W
            xmax = float(obj.findtext("bndbox/xmax")) / W
            ymin = float(obj.findtext("bndbox/ymin")) / H
            ymax = float(obj.findtext("bndbox/ymax")) / H
            box.append([(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin])
        img = Image.open(img_file).convert("RGB")
        if not (self.transform == None):
            img = self.transform(img)
        return img, box, cls

def yoloCollate(batch):
    imgs = []
    targets = []
    for i, (img, box, cls) in enumerate(batch):
        imgs.append(img)
        for j, b in enumerate(box):
            c = cls[j]
            targets.append([i, c] + b)
    imgs = torch.stack(imgs)
    targets = torch.tensor(targets)
    return imgs, targets