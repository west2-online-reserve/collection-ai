# Model validation metrics

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from utils1 import *
from utils import *


# from . import general
#
# def fitness(x):
#     # Model fitness as a weighted combination of metrics
#     w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
#     return (x[:, :4] * w).sum(1)

#计算类别的ap，就是P,R，mAP
def ap_per_class(tp, conf, pred_cls, target_cls):

    # Sort by objectness
    i = np.argsort(-conf) # 将元素从小到大排序，因为加负号,conf大的元素在前面
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i] # 得到的是排序后的信息

    # Find unique classes不一样的类别数，一共就20个类别
    unique_classes = np.unique(target_cls)# 20
    nc = unique_classes.shape[0]  # number of classes, number of detections，20

    # Create Precision-Recall curve and compute AP for each class
    # 画图用的
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    # tp.shape[1]=10，因为map是从0.5到0.95
    # 下面三个都是0，ap.shape=[20,10]
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))

    # 迭代20次，每个类别都要计算相应的ap
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c # 预测值的是c再赋值给i
        n_l = (target_cls == c).sum()  # number of labels 当类别是0的时候，有285个物体
        n_p = i.sum()  # number of predictions 预测出来是类别0的物体，有36935个

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs   计算fp和tp的累计值
            fpc = (1 - tp[i]).cumsum(0)# 计算假正例的累计个数
            tpc = tp[i].cumsum(0)# 计算真正例的累计个数

            # Recall计算召回率
            recall = tpc / (n_l + 1e-16)  # recall curve 看285个物体发现了几个，后面那个值是怕你除0的
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve 求预测曲线，预测正确的个数在所有预测个数的比值
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            #通过recall和precision计算ap，说白了就是算面积，继续迭代，算完第一个类别（类别0的10个iou阈值）
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)#算f1score

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')#把这些值全返回了


def compute_ap(recall, precision):#根据P,R曲线计算ap

    # Append sentinel values to beginning and end赋值操作
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    # 这里是ap计算的一个操作，就是要把ap鼓起来
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)用101个点插值
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate，interp是插值，trapz是积分
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
        # 求积分

    return ap, mpre, mrec
