import torch 
import torchvision

def iou(a, b, xywh = True, eps = 1e-7):
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
    return iou
class ComputemAP:
    def __init__(self, anchors, device, cls_num = 20, iou_threshold = 0.5, conf_threshold = 0.25) -> None:
        self.anchors = torch.tensor(anchors, device = device)
        self.cls_num = cls_num
        self.device = device
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    def Gb(self, x = 20, y = 20):
        res_x, res_y = torch.meshgrid([torch.arange(x), torch.arange(y)])
        res = torch.stack((res_x, res_y), dim=-1).float()
        return res
    
    def __call__(self, p, targets):
        p = self.build_dataset(p)
        prec, rec = self.compute_p_r(p, targets)
        res = self.AP(prec, rec)
        # print(res)
        return res, rec[-1].item() if rec.shape[0] else 0

    def AP(self, prec, rec):
        if not (prec.shape[0] or rec.shape[0]):
            return 0.
        print(prec.shape[0], rec.shape[0])
        ap = 0.
        for i in torch.range(0., 1.1, 0.1):
            if(torch.sum(rec >= i) == 0):
                p = 0
            else:
                p = torch.max(prec[rec >= i])
            ap += p / 11
        return ap

    def compute_p_r(self, p, targets):
        conf, box, cls, img = p
        tp, fp = torch.zeros(conf.shape[0]), torch.zeros(conf.shape[0])
        n = int(conf.shape[0])
        for i in range(n):
            IoU = iou(box[i][None], targets[..., 2:6]).squeeze()
            b = IoU[(targets[..., 0] == img[i]) & (targets[..., 1] == cls[i])]
            # print(b.shape, "QWQ")
            if not b.shape[0]:
                fp[i] = 1
                continue
            mx = b.max(0)[0]
            if(mx >= self.iou_threshold):
                tp[i] = 1
            else:
                fp[i] = 1
        tp = torch.cumsum(tp, 0)
        fp = torch.cumsum(fp, 0)
        # print(tp, fp, "QWQ")
        rec = tp / float(len(targets))
        prec = (tp / (tp + fp)) if (tp.shape[0] and fp.shape[0]) else tp
        # print(rec, prec, "QWQ")
        return prec, rec

    def build_dataset(self, p, nms_threshold = 0.45):
        cf, b, c, imgs = torch.zeros(0, device = self.device), torch.zeros(0, device = self.device), torch.zeros(0, device = self.device), torch.zeros(0, device = self.device)
        for i, pi in enumerate(p):
            batch_size = pi.shape[0]
            # print(pi.shape)
            xy, wh, conf, cls = pi.split((2, 2, 1, self.cls_num), dim = -1)
            conf = conf.squeeze()
            conf = conf[None] if batch_size == 1 else conf
            img = torch.arange(batch_size, device = self.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(conf)
            gxy = self.Gb(pi.shape[2], pi.shape[3]).unsqueeze(0).unsqueeze(0).to(self.device)
            xy = xy.sigmoid() * 2 - 0.5 # (-0.5, 1.5)
            xy = xy + gxy
            wh = (wh.sigmoid() * 2) ** 2 * self.anchors[i].view(1, self.anchors[i].shape[0], 1, 1, self.anchors[i].shape[1])
            box = torch.cat((xy, wh), dim = -1) / pi.shape[2]
            cls_dat, cls = cls.max(dim = -1)
            conf = conf * cls_dat
            box = box.view(-1, 4)
            conf = conf.view(-1)
            cls = cls.view(-1)
            img = img.reshape(-1)
            idx = conf > self.conf_threshold
            if not idx.shape[0]:
                continue
            conf, box, cls, img = conf[idx], box[idx], cls[idx], img[idx]
            idx = conf.argsort(descending = True)
            conf, box, cls, img = conf[idx], box[idx], cls[idx], img[idx]
            # print(conf.shape)
            idx = torchvision.ops.nms(box + cls[..., None], conf, nms_threshold)
            conf, box, cls, img = conf[idx], box[idx], cls[idx], img[idx]
            cf = torch.cat((cf, conf), dim = 0)
            b = torch.cat((b, box), dim = 0)
            c = torch.cat((c, cls), dim = 0)
            imgs = torch.cat((imgs, img), dim = 0)
            # print(cf.shape, conf.shape)
            # print("QWQ")
        return cf, b, c, imgs