import random
import os
import xml

import cv2
import torch


def read_xml_file(filename):
    tree = xml.etree.ElementTree.parse(filename).findall("object")
    objects = []
    for x in tree:
        struct = {}
        # if int(x.find("difficult").text) == 1:
        #     continue
        struct['name'] = x.find("name").text
        struct['box'] = [float(x.find("bndbox").find("xmin").text),
                         float(x.find("bndbox").find("ymin").text),
                         float(x.find("bndbox").find("xmax").text),
                         float(x.find("bndbox").find("ymax").text)]
        objects.append(struct)
    return objects


def process_data():
    # xml_list = os.listdir('C:\\Users\\Classic\\Desktop\\VOCdevkit\\VOC2012\\Annotations')
    xml_list = os.listdir('./VOCdevkit/VOC2012/Annotations')
    random.shuffle(xml_list)
    train_set_size = int(len(xml_list) * 0.8)
    train_list = xml_list[:train_set_size]
    val_list = xml_list[train_set_size:]
    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    file1 = open("train_info.txt", "w")
    file2 = open("val_info.txt", "w")
    for name in train_list:
        # objects = read_xml_file('C:\\Users\\Classic\\Desktop\\VOCdevkit\\VOC2012\\Annotations\\' + name)
        objects = read_xml_file('./VOCdevkit/VOC2012/Annotations/' + name)
        file1.write(name.replace("xml", "jpg"))
        for x in objects:
            file1.write(' ' + str(x['box'][0])
                        + ' ' + str(x['box'][1])
                        + ' ' + str(x['box'][2])
                        + ' ' + str(x['box'][3])
                        + ' ' + str(classes.index(x['name'])))
        file1.write('\n')
    for name in val_list:
        # objects = read_xml_file('C:\\Users\\Classic\\Desktop\\VOCdevkit\\VOC2012\\Annotations\\' + name)
        objects = read_xml_file('./VOCdevkit/VOC2012/Annotations/' + name)
        file2.write(name.replace("xml", "jpg"))
        for x in objects:
            file2.write(' ' + str(x['box'][0])
                        + ' ' + str(x['box'][1])
                        + ' ' + str(x['box'][2])
                        + ' ' + str(x['box'][3])
                        + ' ' + str(classes.index(x['name'])))
        file2.write('\n')
    file1.close()
    file2.close()
    print("process() 执行完毕")


def cal_IoU(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    inter_corner1 = torch.max(boxes1[:, :2], boxes2[:, :2])
    inter_corner2 = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    inter_area = ((inter_corner2[:, 1] - inter_corner1[:, 1]) * (inter_corner2[:, 0] - inter_corner1[:, 0])).clamp(min=0)
    IoU = inter_area / (area1 + area2 - inter_area)
    return IoU.view(-1, 1)


def cal_mAP():
    pass

def detect_example():
    xml_list = os.listdir('C:\\Users\\Classic\\Desktop\\VOCdevkit\\VOC2012\\Annotations')
    # xml_list = os.listdir('./VOCdevkit/VOC2012/Annotations')
    x = random.randint([0,len(xml_list)-1])
    name = 'C:\\Users\\Classic\\Desktop\\VOCdevkit\\VOC2012\\Annotations\\'+ xml_list[x]
    objects = read_xml_file(name)
    cv2.imshow('C:\\Users\\Classic\\Desktop\\VOCdevkit\\VOC2012\\Annotations'+name.replace("xml","jpg"))
