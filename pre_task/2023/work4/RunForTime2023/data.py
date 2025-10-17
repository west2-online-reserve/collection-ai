import random
import torch
import cv2
import numpy


class my_dataset(torch.utils.data.Dataset):
    def __init__(self, path, train, transform):
        super().__init__()
        self.path = path
        self.train = train
        self.transform = transform
        self.filenames = []
        self.labels = []
        self.bounding_boxes = []
        self.sum_boxes = 0
        if train == True:
            lines = open("train_info.txt").readlines()
        else:
            lines = open("val_info.txt").readlines()
        for info in lines:
            temp = info.strip().split()
            self.filenames.append(temp[0])
            boxes_num = (len(temp) - 1) // 5
            self.sum_boxes += boxes_num
            box = []
            label = []
            for i in range(boxes_num):
                box.append(
                    [float(temp[i * 5 + 1]), float(temp[i * 5 + 2]), float(temp[i * 5 + 3]), float(temp[i * 5 + 4])])
                label.append(int(temp[i * 5 + 5]))
            self.bounding_boxes.append(torch.tensor(box))
            self.labels.append(torch.tensor(label))

    def __len__(self):
        return len(self.bounding_boxes)

    def __getitem__(self, index):
        image = cv2.imread(self.path + '/' + self.filenames[index])
        boxes = self.bounding_boxes[index].clone()
        labels = self.labels[index].clone()
        if self.train:
            image, boxes = self.random_horizontal_flip(image, boxes)
            image = self.random_color_jitter(image)
            image = self.random_scale(image)
        height, width, channel = image.shape
        boxes /= torch.tensor([width, height, width, height])  # 转为比例坐标，减少因尺寸变化导致的坐标变换计算
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (448, 448))
        image = self.transform(image)
        target = self.encode(boxes, labels)
        return image, target

    def random_horizontal_flip(self, image, boxes):
        if random.random() < 0.5:
            image_copy = numpy.fliplr(image).copy()
            width = image.shape[1]
            xmin = width - boxes[:, 2]
            xmax = width - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return image_copy, boxes
        return image, boxes

    def random_color_jitter(self, image):  # 调整曝光和饱和度
        image_copy = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image_copy)
        factor1, factor2 = random.choice([0.5, 1.5]), random.choice([0.5, 1.5])
        s = numpy.clip(s * factor1, 0, 255).astype(image_copy.dtype)
        v = numpy.clip(v * factor2, 0, 255).astype(image_copy.dtype)
        image_copy = cv2.merge((h, s, v))
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_HSV2BGR)
        return image_copy

    def random_scale(self,image): # 平滑处理
        if random.random() < 0.5:
            image_copy = cv2.blur(image,(5,5))
            return image_copy
        return image

    def encode(self, boxes, labels):
        target = torch.zeros((7, 7, 30))
        boxes_size = boxes[:, 2:] - boxes[:, :2]
        boxes_center = (boxes[:, :2] + boxes[:, 2:]) / 2
        for i in range(boxes_center.shape[0]):
            grid_cell = (boxes_center[i] * 7).ceil() - 1  # 检测框隶属的网格
            coord_in_grid = boxes_center[i] * 7 - grid_cell  # 目标中心在网格内的比例坐标
            # 检测框的五个参数(x,y,w,h,c)和目标分类。这里设置两个真实检测框的参数一致。
            # 网格的横坐标代表网格位于第几列，同理纵坐标代表第几行。这里存的是行和列
            target[int(grid_cell[1]), int(grid_cell[0]), :2] = coord_in_grid
            target[int(grid_cell[1]), int(grid_cell[0]), 2:4] = boxes_size[i]
            target[int(grid_cell[1]), int(grid_cell[0]), 4] = 1
            target[int(grid_cell[1]), int(grid_cell[0]), 5:7] = coord_in_grid
            target[int(grid_cell[1]), int(grid_cell[0]), 7:9] = boxes_size[i]
            target[int(grid_cell[1]), int(grid_cell[0]), 9] = 1
            target[int(grid_cell[1]), int(grid_cell[0]), int(labels[i]) + 10] = 1
        return target

    def get_sum_boxes(self):
        return self.sum_boxes
