import torch
import math
import torch.nn as nn

def CIOU(a, b, xywh = True, eps = 1e-7): # https://arxiv.org/pdf/1911.08287v1
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = a.chunk(4, -1), b.chunk(4, -1)
        x1_min, x1_max, y1_min, y1_max = x1 - w1 / 2, x1 + w1 / 2, y1 - h1 / 2, y1 + h1 / 2
        x2_min, x2_max, y2_min, y2_max = x2 - w2 / 2, x2 + w2 / 2, y2 - h2 / 2, y2 + h2 / 2
    else:
        (x1_min, x1_max, y1_min, y1_max), (x2_min, x2_max, y2_min, y2_max) = a.chunk(4, -1), b.chunk(4, -1)
        w1, h1 = x1_max - x1_min, (y1_max - y1_min).clamp(eps)
        w2, h2 = x2_max - x2_min, (y2_max - y2_min).clamp(eps)
    I = (x1_max.minimum(x2_max) - x1_min.maximum(x2_min)).clamp(0) * (y1_max.minimum(y2_max) - y1_min.maximum(y2_min)).clamp(0)
    U = w1 * h1 + w2 * h2 - I
    iou = I / U
    cw, ch = x1_max.maximum(x2_max) - x1_min.minimum(x2_min), y1_max.maximum(y2_max) - y1_min.minimum(y2_min)
    rho2 = ((x1_min + x1_max - x2_max - x2_min) ** 2 + (y1_min + y1_max - y2_min - y2_max) ** 2) / 4 # 欧氏距离^2
    c2 = cw ** 2 / ch ** 2 # 覆盖两个box的最小矩形对角线长度
    v = 4 / (math.pi ** 2) * ((torch.arctan(w2 / h2) - torch.arctan(w1 / h1)) ** 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou  - (rho2 / c2 + alpha * v)

class Computeloss:

    def __init__(self, anchors, cls_num = 20, device = "cuda", iou_threshold = 0.1) -> None:
        self.device = device
        self.anchors = torch.tensor(anchors, device = device)
        self.cls_num = cls_num
        self.max_anchor = 4
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(0.5))
        self.BCEconf = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(0.67198))
        self.hyp = {"box" : 0.02, "cls" : 0.21638, "conf" : 0.51728}
        self.iou_threshold = iou_threshold

    def __call__(self, p, targets, compute_mAP = False):
        box_loss = torch.zeros(1, device = self.device)
        cls_loss = torch.zeros(1, device = self.device)
        conf_loss = torch.zeros(1, device = self.device)
        targets_cls, targets_box, targets_indices, anch = self.build_targets(p, targets)

        for i, pi in enumerate(p):
            batch_size = pi.shape[0]
            # print(i, len(targets_indices))
            batch, anchor, gx, gy = targets_indices[i]
            targets_conf = torch.zeros(pi.shape[:4], device = self.device)
            n = batch.shape[0]

            if n:
                # print(pi.shape, batch.shape, anchor.shape, gx.shape, gy.shape)
                xy, wh, _, cls = pi[batch, anchor, gx, gy].split((2, 2, 1, self.cls_num), dim = 1)
                xy = xy.sigmoid() * 2 - 0.5 # (-0.5, 1.5)
                wh = (wh.sigmoid() * 2) ** 2 * anch[i] #先验框是对应好的
                box = torch.cat((xy, wh), dim = 1)
                iou = CIOU(box, targets_box[i]).squeeze()
                box_loss += (1.0 - iou).mean() #box loss
                
                iou = iou.detach().clamp(0)
                targets_conf[batch, anchor, gx, gy] = iou
                t = torch.zeros_like(cls, device = self.device)
                t[range(n), targets_cls[i]] = 1.0
                cls_loss += self.BCEcls(cls, t) # class loss

            conf_loss += self.BCEconf(pi[..., 4], targets_conf)
        box_loss *= self.hyp["box"]
        cls_loss *= self.hyp["cls"]
        conf_loss *= self.hyp["conf"]

        return (box_loss + cls_loss + conf_loss) * batch_size, torch.cat((box_loss, cls_loss, conf_loss)).detach

    def build_targets(self, p, targets): # p 层数(3),batchsize,先验框数(3),x,y,cls + 5; targets num, image + class + box
        anchor_num, targets_num = 3, targets.shape[0]
        target_cls, target_box, target_indices, anch = [], [], [], []
        anchor_id = torch.arange(anchor_num, device = self.device).view(anchor_num, 1).repeat(1, targets_num) # anchor_num,targets_num
        targets = torch.cat((targets.repeat(anchor_num, 1, 1), anchor_id[..., None]), dim = 2) # anchor_num,targets_num, 
        g = 0.5
        xyxy = torch.tensor([[0, 0], [0, 1], [1, 0], [-1, 0], [0, -1]], device = self.device).float() * g
        for i in range(3): # 3层输出
            anchors = self.anchors[i] # 对应先验框
            gain = torch.tensor([1, 1, p[i].shape[3], p[i].shape[2], p[i].shape[3], p[i].shape[2], 1], device = self.device)
            t = targets * gain # 变为当前层的大小
            if targets_num:
                r = t[..., 4:6] / anchors[:, None] # 目标框与先验框的长宽比例
                j = torch.max(r, 1 / r).max(2)[0] < self.max_anchor # 不超过阈值的框
                t = t[j]
                xy = t[:, 2:4]
                nxy = gain[2:4] - xy
                j, k = ((xy % 1 < g) & (xy > 1)).T
                l, m = ((nxy % 1 < g) & (nxy > 1)).T
                j = torch.stack([torch.ones_like(j), m, l, j, k]) # 扩增正样本
                t = t.repeat((5, 1, 1))[j]
                xybias = (torch.zeros_like(xy)[None] + xyxy[:, None])[j] # 对应坐标偏移
            else:
                t = targets[0]
                xybias = 0

            img_and_cls, xy, wh, anchor = t.chunk(4, 1)
            anchor = anchor.long().view(-1) # 每个正样本对应的anchor坐标
            img, cls = img_and_cls.long().T # 图片编号和类别
            gxy = (xy + xybias).long() # 加入偏移后的坐标
            gx, gy = gxy.T

            target_indices.append((img, anchor, gx.clamp_(0, gain[2] - 1), gy.clamp_(0, gain[3] - 1)))
            target_box.append(torch.cat(((xy - gxy), wh), 1)) # 中心点相对网格点的偏移量以及宽高
            anch.append(anchors[anchor]) #每个目标对应的先验框
            target_cls.append(cls)

        return target_cls, target_box, target_indices, anch