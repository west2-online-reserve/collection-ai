import torch
import torch.nn as nn
import torch.nn.functional as F

'''class Criterion_v1(object):
    def __init__(self):
        pass
    
    def box_loss(self, pred_box, true_box):
        pred_xy = pred_box[..., 0:2]
        pred_wh = pred_box[..., 2:4]
        true_xy = true_box[..., 0:2]
        true_wh = true_box[..., 2:4]
        ans = (pred_xy - true_xy) ** 2 # batch_size, 7, 7, 2
        ans += (torch.sqrt(pred_wh) - torch.sqrt(true_wh)) ** 2
        # print(true_wh, "QWQ")
        # print(ans[0, 0, 0], pred_wh[0, 0, 0], true_wh[0, 0, 0],ans.shape, pred_wh.shape, true_wh.shape)
        # while(True):
        #     pass
        ans = ans.sum(dim = 4)
        return ans
    
    def Gb(self, x = 7, y = 7):
        res_x, res_y = torch.meshgrid([torch.arange(x), torch.arange(y)])

        res = torch.stack((res_x, res_y), dim = -1).float()

        return res

    def __call__(self, img, labels, bboxes, device, lam_coord = 5, lam_noobj = 0.5):
        # img = F.sigmoid(img_)
        Gb = self.Gb().unsqueeze(0).unsqueeze(3).to(device)
        pred_bbox = img[..., :10]
        pred_bbox = pred_bbox.reshape(pred_bbox.shape[0], 7, 7, 2, 5)
        pred_bbox[..., 0:2] = (torch.sigmoid(pred_bbox[..., 0:2]) + Gb) / 7.0
        pred_bbox[..., 2:4] = torch.exp(pred_bbox[..., 2:4]) / 7.0
        bboxes = bboxes.unsqueeze(3)
        expand_bboxes = bboxes.expand(-1, -1, -1, 2, -1)
        labeled_boxes = (labels != -1).float()
        labels = torch.nn.functional.one_hot((labels * labeled_boxes).long(), 20).float()
        xmin = torch.max(pred_bbox[..., 0] - pred_bbox[..., 2] / 2, expand_bboxes[..., 0] - expand_bboxes[..., 2] / 2)
        xmax = torch.min(pred_bbox[..., 0] + pred_bbox[..., 2] / 2, expand_bboxes[..., 0] + expand_bboxes[..., 2] / 2)
        ymin = torch.max(pred_bbox[..., 1] - pred_bbox[..., 3] / 2, expand_bboxes[..., 1] - expand_bboxes[..., 3] / 2)
        ymax = torch.min(pred_bbox[..., 1] + pred_bbox[..., 3] / 2, expand_bboxes[..., 1] + expand_bboxes[..., 3] / 2)
        I = torch.clamp((xmax - xmin) * (ymax - ymin), min = 0)
        IoU = I / (pred_bbox[..., 2] * pred_bbox[..., 3] + expand_bboxes[..., 2] * expand_bboxes[..., 3] - I)
        IoU *= labeled_boxes.unsqueeze(-1).expand(-1, -1, -1, 2)

        max_iou, max_iou_id = torch.max(IoU, dim = 3)
        best_boxes = torch.gather(pred_bbox, dim = 3, index = max_iou_id.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 5))
        bad_conf = torch.gather(pred_bbox, dim = 3, index = (max_iou_id ^ 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 5))
        bad_conf = bad_conf[..., -1]
        
        pred_cls = torch.softmax(img[..., 10:], dim = -1)
        
        boxes_loss = lam_coord * self.box_loss(best_boxes, bboxes) * labeled_boxes.unsqueeze(-1)
        conf_loss = ((best_boxes[..., -1].squeeze() - max_iou) * labeled_boxes) ** 2 + lam_noobj * ((best_boxes[..., -1].squeeze() * (labeled_boxes.int() ^ 1)) ** 2 + bad_conf.squeeze() ** 2)
        type_loss = ((pred_cls - labels) * labeled_boxes.unsqueeze(-1)) ** 2

        result = boxes_loss.sum() + conf_loss.sum() + type_loss.sum()
        result = result / img.shape[0]
        return result
    '''
class Match(object):
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def __call__(self, labels, bboxes): #(batch_size, n, *)
        batch_size = labels.shape[0]
        matched_labels = torch.full((batch_size, self.x, self.y), -1, dtype=torch.long)
        matched_bbox = torch.zeros((batch_size, self.x, self.y, 4))
        # print(matched_labels)
        # while(True):
        #     pass
        for i in range(batch_size):
            for label, bbox in zip(labels[i], bboxes[i]):
                if label == -1:
                    continue 
                idx = min(int((bbox[0] * self.x).floor()), self.y - 1) 
                idy = min(int((bbox[1] * self.y).floor()), self.x - 1)
                matched_labels[i][idx][idy] = label
                matched_bbox[i][idx][idy] = bbox

        return matched_labels, matched_bbox

class Criterion_v1(object):
    def __init__(self):
        pass
    
    def box_loss(self, pred_box, true_box):
        pred_xy = pred_box[..., 0:2]
        pred_wh = pred_box[..., 2:4]
        true_xy = true_box[..., 0:2]
        true_wh = true_box[..., 2:4]
        ans = (pred_xy - true_xy) ** 2
        ans += (torch.sqrt(pred_wh) - torch.sqrt(true_wh)) ** 2
        ans = ans.sum(dim=4)
        return ans
    
    def Gb(self, x=7, y=7):
        res_x, res_y = torch.meshgrid([torch.arange(x), torch.arange(y)])
        res = torch.stack((res_x, res_y), dim=-1).float()
        return res

    def __call__(self, img, labels, bboxes, device, lam_coord=10, lam_noobj=0.1, fx = 7, fy = 7):
        Gb = self.Gb(fx, fy).unsqueeze(0).unsqueeze(3).to(device)
        pred_bbox = (img[..., :10])
        pred_bbox = pred_bbox.view(pred_bbox.shape[0], fx, fy, 2, 5)
        pred_bbox[..., 0:2] = (torch.sigmoid(pred_bbox[..., 0:2]) + Gb) / fx
        pred_bbox[..., 2:4] = torch.sigmoid(pred_bbox[..., 2:4])
        # pred_bbox[..., 4] = torch.sigmoid(pred_bbox[..., 4])

        bboxes = bboxes.unsqueeze(3)
        expand_bboxes = bboxes.expand(-1, -1, -1, 2, -1)
        labeled_boxes = (labels != -1).float()
        labels = torch.nn.functional.one_hot((labels * labeled_boxes).long(), 20).float()

        xmin = torch.max(pred_bbox[..., 0] - pred_bbox[..., 2] / 2, expand_bboxes[..., 0] - expand_bboxes[..., 2] / 2)
        xmax = torch.min(pred_bbox[..., 0] + pred_bbox[..., 2] / 2, expand_bboxes[..., 0] + expand_bboxes[..., 2] / 2)
        ymin = torch.max(pred_bbox[..., 1] - pred_bbox[..., 3] / 2, expand_bboxes[..., 1] - expand_bboxes[..., 3] / 2)
        ymax = torch.min(pred_bbox[..., 1] + pred_bbox[..., 3] / 2, expand_bboxes[..., 1] + expand_bboxes[..., 3] / 2)
        # I = torch.clamp((xmax - xmin) * (ymax - ymin), min=0)
        I = (xmax - xmin) * (ymax - ymin)
        I[I < 0] = 0
        IoU = I / (pred_bbox[..., 2] * pred_bbox[..., 3] + expand_bboxes[..., 2] * expand_bboxes[..., 3] - I)
        IoU *= labeled_boxes.unsqueeze(-1).expand(-1, -1, -1, 2)

        # 计算置信度损失
        max_iou, max_iou_id = torch.max(IoU, dim=3)
        best_boxes = torch.gather(pred_bbox, dim=3, index=max_iou_id.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 5))
        bad_boxes = torch.gather(pred_bbox, dim=3, index=(1-max_iou_id).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 5))
        # conf_loss_bad = bad_boxes[..., -1].squeeze() ** 2
        # conf_loss_pos = ((best_boxes[..., -1].squeeze() - max_iou) ** 2) * labeled_boxes
        # conf_loss_neg = ((best_boxes[..., -1].squeeze() * (1 - labeled_boxes)) ** 2)
        conf_loss_neg = F.mse_loss(best_boxes[..., -1].squeeze() * (1 - labeled_boxes), torch.zeros_like(best_boxes[..., -1].squeeze()), size_average=False)
        conf_loss_pos = F.mse_loss(best_boxes[..., -1].squeeze() * labeled_boxes, max_iou * labeled_boxes,size_average=False)
        conf_loss_bad = F.mse_loss(bad_boxes[..., -1].squeeze(), torch.zeros_like(bad_boxes[..., -1].squeeze()), size_average = False)
        conf_loss = (conf_loss_pos.sum() + lam_noobj * (conf_loss_neg.sum() + conf_loss_bad.sum()))

        # 计算类别损失
        pred_cls = img[..., 10:]
        pred_cls = F.softmax(pred_cls, dim = -1)
        type_loss = F.mse_loss(pred_cls * labeled_boxes.unsqueeze(-1), labels * labeled_boxes.unsqueeze(-1),size_average = False)
        # type_loss = ((pred_cls - labels) ** 2 * labeled_boxes.unsqueeze(-1)).sum()

        # 计算边界框损失
        boxes_loss = lam_coord * (self.box_loss(best_boxes, bboxes) * labeled_boxes.unsqueeze(-1)).sum()

        # 计算总损失
        # print(boxes_loss, conf_loss, type_loss)
        result = boxes_loss + conf_loss + type_loss
        result = result / img.shape[0]
        return result
    
class Criterion_v2(object):
    def __init__(self) -> None:
        self.x = 13
        self.y = 13

    def __call__(self, pred, lables, bboxes, device, N, lam_noobj = 0.5, lam_obj = 0.1, lam_class = 0.2, lam_coord = 0.2):
        pred = pred.view(pred.shape[0], -1, N, 25)
        bboxes = bboxes.view(bboxes.shape[0], -1, 4)
        lables = lables.view(lables.shape[0], -1)
        pred_boxes = pred[..., :5]
        pred_cls = pred[..., 5:]
        bboxes = bboxes.unsqueeze(2).expand(-1, -1, N, -1)
        mask_labled = (lables != -1).float()
        lables = F.one_hot((lables * mask_labled).long(), 20).float()
        x_min = torch.max(pred_boxes[..., 0] - pred_boxes[..., 2] / 2., bboxes[..., 0] - bboxes[..., 2] / 2.)
        y_min = torch.max(pred_boxes[..., 1] - pred_boxes[..., 3] / 2., bboxes[..., 1] - bboxes[..., 3] / 2.)
        x_max = torch.min(pred_boxes[..., 0] + pred_boxes[..., 2] / 2., bboxes[..., 0] + bboxes[..., 2] / 2.)
        y_max = torch.min(pred_boxes[..., 1] + pred_boxes[..., 3] / 2., bboxes[..., 1] + bboxes[..., 3] / 2.)
        I = torch.clamp((x_max - x_min) * (y_max - y_min), min = 0).to(device)
        IoU = I / (pred_boxes[..., 2] * pred_boxes[..., 3] + bboxes[..., 2] * bboxes[..., 3] - I)
        # print(pred_boxes.shape)
        unlabled_box = pred_boxes * (1 - mask_labled.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, 5))
        max_iou, max_iou_id = torch.max(IoU, dim = -1)
        max_iou_id = F.one_hot(max_iou_id, 5).float().unsqueeze(-1).expand(-1, -1, -1, 5)
        targeted_box = pred_boxes[max_iou_id == 1].squeeze()
        no_targeted_box = pred_boxes[max_iou_id != 1]
        max_iou_id = max_iou_id[..., 0].unsqueeze(-1).expand(-1, -1, -1, 20)
        targeted_cls = pred_cls[max_iou_id == 1].view(pred_cls.shape[0], -1, 20)
        # no_targeted_cls = pred_cls[max_iou_id != 1]
        
        # print(pred_cls.shape, max_iou_id.shape, targeted_cls.shape)
        # print(lables.shape)
        conf_loss = lam_noobj * (((no_targeted_box[..., 4] * mask_labled.unsqueeze(-1) - 0) ** 2).sum() + ((unlabled_box[..., 4] - 0) ** 2).sum()) + lam_obj * (((targeted_box[..., 4] - max_iou) * mask_labled) ** 2).sum()
        cls_loss = lam_class * ((targeted_cls - lables) * mask_labled.unsqueeze(-1)) ** 2
        box_loss1 = lam_coord * ((2 - targeted_box[..., 2] * targeted_box[..., 3]).unsqueeze(-1) * mask_labled.unsqueeze(-1) * (targeted_box[..., :4] - bboxes[..., 0, :4]) ** 2)
        # bboxes.
        box_loss2 = 0.01 * ((no_targeted_box[..., :4] - bboxes[..., :4, :4]) * mask_labled.unsqueeze(-1).unsqueeze(-1)) ** 2
        # print(conf_loss.sum() , cls_loss.sum() , box_loss1.sum() , box_loss2.sum())
        # while True:
        #     pass
        ans = conf_loss.sum() + cls_loss.sum() + box_loss1.sum() + box_loss2.sum()
        return ans