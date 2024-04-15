import random
import torch
import cv2
import numpy


class myDataset(torch.utils.data.Dataset):
    def __init__(self, path, train, transform):
        super().__init__()
        # self.path = path + "\\VOCdevkit\\VOC2012\\JPEGImages"
        self.path = path + "VOCdevkit/VOC2012/JPEGImages"
        self.train = train
        self.transform = transform
        self.filenames = []
        self.labels = []
        self.bounding_boxes = []
        if train == True:
            lines = open("train_info.txt").readlines()
        else:
            lines = open("val_info.txt").readlines()
        for info in lines:
            temp = info.strip().split()
            self.filenames.append(temp[0])
            num_boxes = (len(temp) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                xmin = float(temp[i * 5 + 1])
                ymin = float(temp[i * 5 + 2])
                xmax = float(temp[i * 5 + 3])
                ymax = float(temp[i * 5 + 4])
                tag = int(temp[i * 5 + 5])
                box.append([xmin, ymin, xmax, ymax])
                label.append(tag)
            self.bounding_boxes.append(torch.tensor(box))
            self.labels.append(torch.tensor(label))

    def __len__(self):
        return len(self.bounding_boxes)

    def __getitem__(self, index):
        # image = cv2.imread(self.path + '\\' + self.filenames[index])
        image = cv2.imread(self.path + '/' + self.filenames[index])
        boxes = self.bounding_boxes[index].clone()
        labels = self.labels[index].clone()
        if self.train:
            image, boxes = self.random_horizontal_flip(image, boxes)
        h, w, c = image.shape
        boxes /= torch.tensor([w, h, w, h])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (448, 448))
        image = self.transform(image)
        target = self.encoder(boxes, labels)
        return image, target

    def random_horizontal_flip(self, image, boxes):
        if random.random() < 0.5:
            image_copy = numpy.fliplr(image).copy()
            w = image.shape[1]
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return image_copy, boxes
        return image, boxes

    def encoder(self, boxes, labels):
        target = torch.zeros((7, 7, 30))
        cell_size = 1.0 / 7
        w_and_h = boxes[:, 2:] - boxes[:, :2]
        # 物体中心坐标
        centers = (boxes[:, 2:] + boxes[:, :2]) / 2
        for i in range(centers.size()[0]):
            corner = (centers[i] / cell_size).ceil() - 1  # 网格左上角坐标
            # 两个框的置信度
            target[int(corner[1]), int(corner[0]), 4] = 1
            target[int(corner[1]), int(corner[0]), 9] = 1
            target[int(corner[1]), int(corner[0]), int(labels[i]) + 10] = 1

            ratio = corner * cell_size  # 变为比例坐标
            delta = (centers[i] - ratio) / cell_size

            target[int(corner[1]), int(corner[0]), :2] = delta
            target[int(corner[1]), int(corner[0]), 2:4] = w_and_h[i]
            target[int(corner[1]), int(corner[0]), 5:7] = delta
            target[int(corner[1]), int(corner[0]), 7:9] = w_and_h[i]
        return target