import torch
import numpy as np
import matplotlib.pyplot as plt

def IoU(x, y):
    I = (min(x[0] + x[2] / 2, y[0] + y[2] / 2) - max(x[0] - x[2] / 2, y[0] - y[2] / 2)) * (min(x[1] + x[3] / 2, y[1] + y[3] / 2) - max(x[1] - x[3] / 2, y[1] - y[3] / 2))
    U = x[2] * x[3] + y[2] * y[3] - I
    return I * 1.0 / U

def AP(rec, prec): # VOC2007中使用的AP计算方法(输入为recall和precision)
    #11个recall阈值
    ap = 0.
    for i in torch.range(0., 1.1, 0.1):
        if(torch.sum(rec >= i) == 0):
            p = 0
        else:
            p = torch.max(prec[rec >= i])
        ap += p / 11.
    return ap

def mAP(targets, cnt, device, IoU_threshold = 0.5):
    mAP = 0.
    sum_tp = 0
    sum_cnt = 0
    rec = None
    for i, target in enumerate(targets):
        sum_cnt += cnt[i]
        # print(target)
        target = torch.tensor([it.cpu().detach().numpy() for it in target]).to(device)
        if(target.shape[0] == 0):
            continue
        # print(target.shape)
        _, sorted_id = torch.sort(target[:, 4], descending=True)
        target = target[sorted_id]
        n = len(target)
        tp = torch.zeros(n)
        fp = torch.zeros(n)
        for j in range(n):
            if(target[j][-1] == -1):
                fp[j] = 1
                continue
            if(IoU(target[j][0:4], target[j][5:9]) >= IoU_threshold):
                tp[j] = 1
            else:
                fp[j] = 1
        tp = torch.cumsum(tp, 0)
        fp = torch.cumsum(fp, 0)
        rec = tp / float(cnt[i])
        prec = tp / (tp + fp)
        ap = AP(rec, prec)
        mAP += ap
        sum_tp += tp[-1] if len(tp) != 0 else 0
    return mAP / 20., sum_tp / float(sum_cnt) if sum_cnt != 0 else 0

        