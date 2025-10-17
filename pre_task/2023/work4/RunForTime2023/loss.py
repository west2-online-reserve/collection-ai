import torch
import torchvision
# from utils import cal_IoU


class custom_loss(torch.nn.Module):
    def __init__(self, coord, noobj):
        super().__init__()
        self.coord = coord
        self.noobj = noobj

    def forward(self, output, target):
        MSE = torch.nn.MSELoss(reduction='sum')  # 返回总损失
        with_object = target[:, :, :, 4] > 0  # 含检测目标的网格
        without_object = target[:, :, :, 4] == 0
        with_object = with_object.unsqueeze(-1).expand_as(target)
        without_object = without_object.unsqueeze(-1).expand_as(target)
        output_with_obj = output[with_object].view(-1, 30)
        target_with_obj = target[with_object].view(-1, 30)
        output_boxes = output_with_obj[:, :10].contiguous().view(-1, 5)
        output_classes = output_with_obj[:, 10:]
        target_boxes = target_with_obj[:, :10].contiguous().view(-1, 5)
        target_classes = target_with_obj[:, 10:]
        class_loss = MSE(output_classes, target_classes)

        temp1, index1 = torch.max(output_classes, dim=1)
        temp2, index2 = torch.max(target_classes, dim=1)
        accurate_num = torch.sum(index1 == index2).item()

        left = torch.zeros(target_boxes.size(), dtype=torch.bool)  # 交并比更大的检测框留下，计入1_ij(obj)
        abandon = torch.zeros(target_boxes.size(), dtype=torch.bool)
        box_iou = torch.zeros(target_boxes.size())
        # left = torch.cuda.BoolTensor(target_boxes.size()).zero_()
        # abandon = torch.cuda.BoolTensor(target_boxes.size()).zero_()
        # box_iou = torch.cuda.FloatTensor(target_boxes.size()).zero_()
        sum_iou = 0
        for i in range(0, target_boxes.shape[0], 2):
            box1 = output_boxes[i:i + 2]
            box2 = target_boxes[i].view(-1, 5)  # 变为1*5的张量
            output_xy = torch.autograd.Variable(torch.zeros(box1.size(), dtype=torch.float32, requires_grad=True))
            target_xy = torch.autograd.Variable(torch.zeros(box2.size(), dtype=torch.float32, requires_grad=True))
            # output_xy = torch.autograd.Variable(torch.cuda.FloatTensor(box1.size()))
            # target_xy = torch.autograd.Variable(torch.cuda.FloatTensor(box2.size()))
            # (x,y,w,h)-->(xmin,ymin,xmax,ymax)
            output_xy[:, :2] = box1[:, :2] / 7 - 0.5 * box1[:, 2:4]
            output_xy[:, 2:4] = box1[:, :2] / 7 + 0.5 * box1[:, 2:4]
            target_xy[:, :2] = box2[:, :2] / 7 - 0.5 * box2[:, 2:4]
            target_xy[:, 2:4] = box2[:, :2] / 7 + 0.5 * box2[:, 2:4]
            # iou = cal_IoU(output_xy[:, :4], target_xy[:, :4])
            iou = torchvision.ops.box_iou(output_xy[:,:4],target_xy[:,:4])
            # max_iou, index = torch.max(iou), torch.argmax(iou)
            max_iou, index = iou.max(0)
            left[i + index], abandon[i + 1 - index] = True, True
            box_iou[i + index, 4] = max_iou  # 以IoU作为置信度
            sum_iou += max_iou.item()

        box_iou = torch.autograd.Variable(box_iou)  # 置信度损失的梯度与IoU的计算方式无关
        output_boxes_left = output_boxes[left].view(-1, 5)
        target_boxes_left_coord = target_boxes[left].view(-1, 5)
        target_boxes_left_IoU = box_iou[left].view(-1, 5)
        output_boxes_abandon = output_boxes[abandon].view(-1, 5)
        target_boxes_abandon_IoU = box_iou[abandon].view(-1, 5)

        left_boxes_coord_loss = MSE(output_boxes_left[:, :2], target_boxes_left_coord[:, :2]) + MSE(
            torch.sqrt(output_boxes_left[:, 2:4]), torch.sqrt(target_boxes_left_coord[:, 2:4]))
        left_boxes_confidence_loss = MSE(output_boxes_left[:, 4], target_boxes_left_IoU[:, 4])
        abandon_boxes_loss = MSE(output_boxes_abandon[:, 4], target_boxes_abandon_IoU[:, 4])  # 计入1_ij(noobj)

        output_without_obj = output[without_object].view(-1, 30)
        target_without_obj = target[without_object].view(-1, 30)
        temp1 = torch.zeros(output_without_obj.size(), dtype=torch.bool)
        # temp1 = torch.cuda.BoolTensor(output_without_obj.size()).zero_()
        temp1[:, 4], temp1[:, 9] = True, True
        noobj_output_c = output_without_obj[temp1]
        noobj_target_c = target_without_obj[temp1]
        noobj_loss = MSE(noobj_output_c, noobj_target_c)

        return self.coord * left_boxes_coord_loss + left_boxes_confidence_loss + self.noobj * (
                noobj_loss + abandon_boxes_loss) + class_loss, sum_iou, accurate_num
