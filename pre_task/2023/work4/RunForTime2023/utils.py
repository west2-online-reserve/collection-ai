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
