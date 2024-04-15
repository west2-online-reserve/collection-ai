import xml.etree.ElementTree as ET
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches


basedir = "./data/train_data/VOCdevkit/VOC2012"
class_code = {"person": 0, "bird": 1, "cat": 2, "cow": 3, "dog": 4, "horse": 5, "sheep": 6, "aeroplane": 7,
              "bicycle": 8, "boat": 9, "bus": 10, "car": 11, "motorbike": 12, "train": 13, "bottle": 14,
              "chair": 15, "diningtable": 16, "pottedplant": 17, "sofa": 18, "tvmonitor": 19}
# feature_size = [224, 224]

def read_train():
    listfilename = os.path.join(basedir, "ImageSets", "Segmentation", "train.txt")
    listname = []
    bbox = []
    label = []
    img = []
    with open(listfilename) as f:
        listname = f.readlines()
    for i, name in enumerate(listname):
        xml_file = os.path.join(basedir, "Annotations", name[:-1] + ".xml")
        img_file = os.path.join(basedir, "JPEGImages", name[:-1] + ".jpg")
        tree = ET.parse(xml_file)
        img_data = Image.open(img_file).convert("RGB")
        W = float(tree.findtext("size/width"))
        H = float(tree.findtext("size/height"))
        label.append([])
        bbox.append([])
        # fig, ax = plt.subplots()
        # ax.imshow(img_data)
        for obj in tree.iter("object"):
            label[i].append(class_code[obj.findtext("name")])
            box = []
            xmin = float(obj.findtext("bndbox/xmin")) / W
            ymin = float(obj.findtext("bndbox/ymin")) / H
            xmax = float(obj.findtext("bndbox/xmax")) / W
            ymax = float(obj.findtext("bndbox/ymax")) / H
            box.append((xmin + xmax) / 2)
            box.append((ymin + ymax) / 2)
            box.append(xmax - xmin)
            box.append(ymax - ymin)
            bbox[i].append(box)
            # boxs = patches.Rectangle((box[0] * W, box[1] * H), box[2] * W, box[3] * H, linewidth = 2, edgecolor = 'r', facecolor = 'none')
            # ax.add_patch(boxs)
            # plt.text(box[0] * W, box[1] * H, obj.findtext("name"), color= "red")
        # plt.axis('off')
        # plt.show()
        img.append(img_data)
    return img, label, bbox

def read_val():
    listfilename = os.path.join(basedir, "ImageSets", "Segmentation", "val.txt")
    listname = []
    bbox = []
    label = []
    img = []
    with open(listfilename) as f:
        listname = f.readlines()
    for i, name in enumerate(listname):
        xml_file = os.path.join(basedir, "Annotations", name[:-1] + ".xml")
        img_file = os.path.join(basedir, "JPEGImages", name[:-1] + ".jpg")
        tree = ET.parse(xml_file)
        img_data = Image.open(img_file).convert("RGB")
        W = float(tree.findtext("size/width"))
        H = float(tree.findtext("size/height"))
        label.append([])
        bbox.append([])
        for obj in tree.iter("object"):
            label[i].append(class_code[obj.findtext("name")])
            box = [] # x, y, w, h
            xmin = float(obj.findtext("bndbox/xmin")) / W
            ymin = float(obj.findtext("bndbox/ymin")) / H
            xmax = float(obj.findtext("bndbox/xmax")) / W
            ymax = float(obj.findtext("bndbox/ymax")) / H
            box.append((xmin + xmax) / 2)
            box.append((ymin + ymax) / 2)
            box.append(xmax - xmin)
            box.append(ymax - ymin)
            bbox[i].append(box)
        img.append(img_data)
    return img, label, bbox

def cmp(batch):
    imgs = []
    labels = []
    bboxs = []
    for item in batch:
        img, label, bbox = item
        imgs.append(img)
        labels.append(torch.tensor(label))
        bboxs.append(torch.tensor(bbox))
    imgs = torch.stack(imgs)
    bboxs = torch.nn.utils.rnn.pad_sequence(bboxs, batch_first=True, padding_value = 0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value = -1)
    return imgs, labels, bboxs

    