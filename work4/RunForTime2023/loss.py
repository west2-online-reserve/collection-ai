import torch


class custom_loss(torch.nn.Module):
    def __init__(self, S, B, coord, noobj, device):
        super().__init__()
        self.S = S
        self.B = B
        self.coord = coord
        self.noobj = noobj
        self.device = device

    def cal_IoU(self, boxes1, boxes2):
        areas1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        areas2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        # inter_upperlefts,inter_lowerrights,inters的形状:
        # (boxes1的数量,boxes2的数量,2)
        inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
        # inter_area,union_area的形状:(boxes1的数量,boxes2的数量)
        inter_area = inters[:, :, 0] * inters[:, :, 1]
        union_area = areas1[:, None] + areas2 - inter_area
        return inter_area / union_area

    def forward(self, output, target):
        batch_size = output.size()[0]
        MSE = torch.nn.MSELoss()
        t1 = target[:, :, :, 4] > 0  # 需要预测bounding_box的
        t2 = target[:, :, :, 4] == 0  # 不需要预测bounding_box的
        t1 = t1.unsqueeze(-1).expand_as(target)
        t2 = t2.unsqueeze(-1).expand_as(target)
        output_with_obj = output[t1].view(-1, 30)
        target_with_obj = target[t1].view(-1, 30)
        output_box = output_with_obj[:, :10].contiguous().view(-1, 5)
        output_class = output_with_obj[:, 10:]
        target_box = target_with_obj[:, :10].contiguous().view(-1, 5)
        target_class = target_with_obj[:, 10:]

        temp1, index1 = torch.max(output_class, dim=1)
        temp2, index2 = torch.max(target_class, dim=1)
        comp = index1 == index2
        accurateNum = torch.sum(comp)

        t3 = torch.zeros(target_box.size(), dtype=torch.bool, device=self.device)
        t7 = torch.zeros(target_box.size(), dtype=torch.bool, device=self.device)
        box_IoU = torch.zeros(target_box.size(), device=self.device)
        sum_iou = 0

        for i in range(0, target_box.size()[0], 2):
            box1 = output_box[i:i + 2]
            box2 = target_box[i].view(-1, 5)
            t5 = torch.autograd.Variable(torch.zeros(box1.size(), dtype=torch.float, device=self.device))
            t6 = torch.autograd.Variable(torch.zeros(box2.size(), dtype=torch.float, device=self.device))
            t5[:, :2] = box1[:, :2] / self.S - 0.5 * box1[:, 2:4]
            t5[:, 2:4] = box1[:, :2] / self.S + 0.5 * box1[:, 2:4]
            t6[:, :2] = box2[:, :2] / self.S - 0.5 * box2[:, 2:4]
            t6[:, 2:4] = box2[:, :2] / self.S + 0.5 * box2[:, 2:4]
            IoU = self.cal_IoU(t5[:, :4], t6[:, :4])
            max_iou, max_index = IoU.max(0)
            max_index = max_index.to(self.device)
            t3[i + max_index] = 1  # IoU较大的 bounding box
            t7[i + 1 - max_index] = 1  # 舍去的 bounding box
            # 计算置信度
            box_IoU[i + max_index, 4] = max_iou.to(self.device)
            sum_iou += IoU.min(0)[0]

        box_IoU = torch.autograd.Variable(box_IoU)
        # 置信度误差（含物体的grid ceil的两个bounding_box与ground truth的“交并比”较大的一方）
        output_box_response = output_box[t3].view(-1, 5)
        target_box_response_IoU = box_IoU[t3].view(-1, 5)
        # “交并比”较小的一方
        no_box_pred_response = output_box[t7].view(-1, 5)
        no_box_target_response_IoU = box_IoU[t7].view(-1, 5)
        no_box_target_response_IoU[:, 4] = 0

        target_box_response = target_box[t3].view(-1, 5)

        contain_box_loss = MSE(output_box_response[:, :2], target_box_response[:, :2]) + MSE(
            torch.sqrt(output_box_response[:, 2:4]), torch.sqrt(target_box_response[:, 2:4]))
        not_contain_box_loss = MSE(no_box_pred_response[:, 4], no_box_target_response_IoU[:, 4])
        obj_loss = MSE(output_box_response[:, 4], target_box_response_IoU[:, 4])

        output_without_obj = output[t2].view(-1, 30)
        target_without_obj = target[t2].view(-1, 30)
        t4 = torch.zeros(output_without_obj.size(), dtype=torch.bool, device=self.device)
        t4[:, 4] = 1
        t4[:, 9] = 1
        output_without_obj_C_hat = output_without_obj[t4]
        target_without_obj_C_hat = target_without_obj[t4]
        noobj_loss = MSE(output_without_obj_C_hat, target_without_obj_C_hat)

        class_loss = MSE(output_class, target_class)

        return self.coord * contain_box_loss + obj_loss + self.noobj * (
                    noobj_loss + not_contain_box_loss) + class_loss, sum_iou, accurateNum
