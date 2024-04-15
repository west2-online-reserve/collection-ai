import os
import glob
import json
import sys
import cv2
import math
import datetime
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.signal
import shutil
import torch.optim as optim
import operator
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from copy import deepcopy
from functools import partial
from torchvision.ops import nms
from PIL import Image
from tqdm import tqdm

from utils import (get_classes,yolo_dataset_collate,make_anchors,select_candidates_in_gts,select_highest_overlaps,bbox_iou,bbox2dist,bbox_iou,
                    de_parallel,is_parallel,cvtColor,resize_image,get_map,dist2bbox,file_lines_to_list,draw_text_in_image,log_average_miss_rate,
                    draw_plot_func,xywh2xyxy,preprocess_input)

class_all = (
    "DecodeBox"
    "EvalCallback",
    "BboxLoss",
    "TaskAlignedAssigner",
    "Loss",
    "ModelEMA",
    "LossHistory",
    "BboxLoss",
)

class DecodeBox():
    def __init__(self, num_classes, input_shape):
        super(DecodeBox, self).__init__()
        self.num_classes    = num_classes
        self.bbox_attrs     = 4 + num_classes
        self.input_shape    = input_shape
        
    def decode_box(self, inputs):
        # dbox  batch_size, 4, 8400
        # cls   batch_size, 20, 8400
        dbox, cls, origin_cls, anchors, strides = inputs
        # 获得中心宽高坐标
        dbox    = dist2bbox(dbox, anchors.unsqueeze(0), xywh=True, dim=1) * strides
        y       = torch.cat((dbox, cls.sigmoid()), 1).permute(0, 2, 1)
        # 进行归一化，到0~1之间
        y[:, :, :4] = y[:, :, :4] / torch.Tensor([self.input_shape[1], self.input_shape[0], self.input_shape[1], self.input_shape[0]]).to(y.device)
        return y

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            #-----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            #-----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
        #----------------------------------------------------------#
        #   将预测结果的格式转换成左上角右下角的格式。
        #   prediction  [batch_size, num_anchors, 85]
        #----------------------------------------------------------#
        box_corner          = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            #----------------------------------------------------------#
            #   对种类预测部分取max。
            #   class_conf  [num_anchors, 1]    种类置信度
            #   class_pred  [num_anchors, 1]    种类
            #----------------------------------------------------------#
            class_conf, class_pred = torch.max(image_pred[:, 4:4 + num_classes], 1, keepdim=True)

            #----------------------------------------------------------#
            #   利用置信度进行第一轮筛选
            #----------------------------------------------------------#
            conf_mask = (class_conf[:, 0] >= conf_thres).squeeze()
            
            #----------------------------------------------------------#
            #   根据置信度进行预测结果的筛选
            #----------------------------------------------------------#
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.size(0):
                continue
            #-------------------------------------------------------------------------#
            #   detections  [num_anchors, 6]
            #   6的内容为：x1, y1, x2, y2, class_conf, class_pred
            #-------------------------------------------------------------------------#
            detections = torch.cat((image_pred[:, :4], class_conf.float(), class_pred.float()), 1)

            #------------------------------------------#
            #   获得预测结果中包含的所有种类
            #------------------------------------------#
            unique_labels = detections[:, -1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            for c in unique_labels:
                #------------------------------------------#
                #   获得某一类得分筛选后全部的预测结果
                #------------------------------------------#
                detections_class = detections[detections[:, -1] == c]
                #------------------------------------------#
                #   使用官方自带的非极大抑制会速度更快一些！
                #   筛选出一定区域内，属于同一种类得分最大的框
                #------------------------------------------#
                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4],
                    nms_thres
                )
                max_detections = detections_class[keep]
                
                # # 按照存在物体的置信度排序
                # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
                # detections_class = detections_class[conf_sort_index]
                # # 进行非极大抑制
                # max_detections = []
                # while detections_class.size(0):
                #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
                #     max_detections.append(detections_class[0].unsqueeze(0))
                #     if len(detections_class) == 1:
                #         break
                #     ious = bbox_iou(max_detections[-1], detections_class[1:])
                #     print(ious)
                #     detections_class = detections_class[1:][ious < nms_thres]
                # # 堆叠
                # max_detections = torch.cat(max_detections).data
                
                # Add max detections to outputs
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
            
            if output[i] is not None:
                output[i]           = output[i].cpu().numpy()
                box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4]    = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output

class EvalCallback():
    def __init__(self, net, input_shape, class_names, num_classes, val_lines, log_dir, cuda, \
            map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, nms_iou=0.5, letterbox_image=True, MINOVERLAP=0.5, eval_flag=True, period=1):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.class_names        = class_names
        self.num_classes        = num_classes
        self.val_lines          = val_lines
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.map_out_path       = map_out_path
        self.max_boxes          = max_boxes
        self.confidence         = confidence
        self.nms_iou            = nms_iou
        self.letterbox_image    = letterbox_image
        self.MINOVERLAP         = MINOVERLAP
        self.eval_flag          = eval_flag
        self.period             = period
        
        self.bbox_util          = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]))
        
        self.maps       = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

        top_100     = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes   = top_boxes[top_100]
        top_conf    = top_conf[top_100]
        top_label   = top_label[top_100]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
    
    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))
            print("Get map.")
            for annotation_line in tqdm(self.val_lines):
                line        = annotation_line.split()
                image_id    = os.path.basename(line[0]).split('.')[0]
                #------------------------------#
                #   读取图像并转换成RGB图像
                #------------------------------#
                image       = Image.open(line[0])
                #------------------------------#
                #   获得预测框
                #------------------------------#
                gt_boxes    = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
                #------------------------------#
                #   获得预测txt
                #------------------------------#
                self.get_map_txt(image_id, image, self.class_names, self.map_out_path)
                
                #------------------------------#
                #   获得真实框txt
                #------------------------------#
                with open(os.path.join(self.map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
                        
            print("Calculate Map.")
            temp_map = get_map(self.MINOVERLAP, True, path = self.map_out_path)
            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth = 2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s'%str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            shutil.rmtree(self.map_out_path)

class BboxLoss(nn.Module):
    def __init__(self, reg_max=16, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # 计算IOU损失
        # weight代表损失中标签应该有的置信度，0最小，1最大
        weight      = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        # 计算预测框和真实框的重合程度
        iou         = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        # print(iou)
        # 然后1-重合程度，乘上应该有的置信度，求和后求平均。
        loss_iou    = ((1.0 - iou) * weight).sum() / target_scores_sum

        # 计算DFL损失
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        # 一个点一般不会处于anchor点上，一般是xx.xx。如果要用DFL的话，不可能直接一个cross_entropy就能拟合
        # 所以把它认为是相对于xx.xx左上角锚点与右下角锚点的距离 如果距离右下角锚点距离小，wl就小，左上角损失就小
        #                                                   如果距离左上角锚点距离小，wr就小，右下角损失就小
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)

class TaskAlignedAssigner(nn.Module):

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9, roll_out_thr=0):
        super().__init__()
        self.topk           = topk
        self.num_classes    = num_classes
        self.bg_idx         = num_classes
        self.alpha          = alpha
        self.beta           = beta
        self.eps            = eps
        # roll_out_thr为64
        self.roll_out_thr   = roll_out_thr

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):

        # 获得batch_size 
        self.bs             = pd_scores.size(0)
        # 获得真实框中的最大框数量
        self.n_max_boxes    = gt_bboxes.size(1)
        # 如果self.n_max_boxes大于self.roll_out_thr则roll_out
        self.roll_out       = self.n_max_boxes > self.roll_out_thr if self.roll_out_thr else False
    
        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        # b, max_num_obj, 8400
        # mask_pos      满足在真实框内、是真实框topk最重合的正样本、满足mask_gt的锚点
        # align_metric  某个先验点属于某个真实框的类的概率乘上某个先验点与真实框的重合程度
        # overlaps      所有真实框和锚点的重合程度
        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)

        # target_gt_idx     b, 8400     每个anchor符合哪个gt
        # fg_mask           b, 8400     每个anchor是否有符合的gt
        # mask_pos          b, max_num_obj, 8400    one_hot后的target_gt_idx
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # 指定目标到对应的anchor点上
        # b, 8400
        # b, 8400, 4
        # b, 8400, 80
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # 乘上mask_pos，把不满足真实框满足的锚点的都置0
        align_metric        *= mask_pos
        # 每个真实框对应的最大得分
        # b, max_num_obj
        pos_align_metrics   = align_metric.amax(axis=-1, keepdim=True) 
        # 每个真实框对应的最大重合度
        # b, max_num_obj
        pos_overlaps        = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
        # 把每个真实框和先验点的得分乘上最大重合程度，再除上最大得分
        norm_align_metric   = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        # target_scores作为正则的标签
        target_scores       = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        # pd_scores bs, num_total_anchors, num_classes
        # pd_bboxes bs, num_total_anchors, 4
        # gt_labels bs, n_max_boxes, 1
        # gt_bboxes bs, n_max_boxes, 4
        # 
        # align_metric是一个算出来的代价值，某个先验点属于某个真实框的类的概率乘上某个先验点与真实框的重合程度
        # overlaps是某个先验点与真实框的重合程度
        # align_metric, overlaps    bs, max_num_obj, 8400
        align_metric, overlaps  = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        
        # 正样本锚点需要同时满足：
        # 1、在真实框内
        # 2、是真实框topk最重合的正样本
        # 3、满足mask_gt
        
        # get in_gts mask           b, max_num_obj, 8400
        # 判断先验点是否在真实框内
        mask_in_gts             = select_candidates_in_gts(anc_points, gt_bboxes, roll_out=self.roll_out)
        # get topk_metric mask      b, max_num_obj, 8400
        # 判断锚点是否在真实框的topk中
        mask_topk               = self.select_topk_candidates(align_metric * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        # merge all mask to a final mask, b, max_num_obj, h*w
        # 真实框存在，非padding
        mask_pos                = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        if self.roll_out:
            align_metric    = torch.empty((self.bs, self.n_max_boxes, pd_scores.shape[1]), device=pd_scores.device)
            overlaps        = torch.empty((self.bs, self.n_max_boxes, pd_scores.shape[1]), device=pd_scores.device)
            ind_0           = torch.empty(self.n_max_boxes, dtype=torch.long)
            for b in range(self.bs):
                ind_0[:], ind_2 = b, gt_labels[b].squeeze(-1).long()
                # 获得属于这个类别的得分
                # bs, max_num_obj, 8400
                bbox_scores     = pd_scores[ind_0, :, ind_2]  
                # 计算真实框和预测框的ciou
                # bs, max_num_obj, 8400
                overlaps[b]     = bbox_iou(gt_bboxes[b].unsqueeze(1), pd_bboxes[b].unsqueeze(0), xywh=False, CIoU=True).squeeze(2).clamp(0)
                align_metric[b] = bbox_scores.pow(self.alpha) * overlaps[b].pow(self.beta)
        else:
            # 2, b, max_num_obj
            ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)       
            # b, max_num_obj  
            # [0]代表第几个图片的
            ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)  
            # [1]真是标签是什么
            ind[1] = gt_labels.long().squeeze(-1) 
            # 获得属于这个类别的得分
            # 取出某个先验点属于某个类的概率
            # b, max_num_obj, 8400
            bbox_scores = pd_scores[ind[0], :, ind[1]]  

            # 计算真实框和预测框的ciou
            # bs, max_num_obj, 8400
            overlaps        = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False, CIoU=True).squeeze(3).clamp(0)
            align_metric    = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):

        # 8400
        num_anchors             = metrics.shape[-1] 
        # b, max_num_obj, topk
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        # b, max_num_obj, topk
        topk_idxs[~topk_mask] = 0
        # b, max_num_obj, topk, 8400 -> b, max_num_obj, 8400
        # 这一步得到的is_in_topk为b, max_num_obj, 8400
        # 代表每个真实框对应的top k个先验点
        if self.roll_out:
            is_in_topk = torch.empty(metrics.shape, dtype=torch.long, device=metrics.device)
            for b in range(len(topk_idxs)):
                is_in_topk[b] = F.one_hot(topk_idxs[b], num_anchors).sum(-2)
        else:
            is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)
        # 判断锚点是否在真实框的topk中
        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):

        # 用于读取真实框标签, (b, 1)
        batch_ind       = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        # b, h*w    获得gt_labels，gt_bboxes在flatten后的序号
        target_gt_idx   = target_gt_idx + batch_ind * self.n_max_boxes
        # b, h*w    用于flatten后读取标签
        target_labels   = gt_labels.long().flatten()[target_gt_idx]
        # b, h*w, 4 用于flatten后读取box
        target_bboxes   = gt_bboxes.view(-1, 4)[target_gt_idx]
        
        # assigned target scores
        target_labels.clamp(0)
        # 进行one_hot映射到训练需要的形式。
        target_scores   = F.one_hot(target_labels, self.num_classes)  # (b, h*w, 80)
        fg_scores_mask  = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores   = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

class Loss:
    def __init__(self, model): 
        self.bce    = nn.BCEWithLogitsLoss(reduction='none')
        self.stride = model.stride  # model strides
        self.nc     = model.num_classes  # number of classes
        self.no     = model.no
        self.reg_max = model.reg_max
        
        self.use_dfl = model.reg_max > 1
        roll_out_thr = 64

        self.assigner = TaskAlignedAssigner(topk=10,
                                            num_classes=self.nc,
                                            alpha=0.5,
                                            beta=6.0,
                                            roll_out_thr=roll_out_thr)
        self.bbox_loss  = BboxLoss(model.reg_max - 1, use_dfl=self.use_dfl)
        self.proj       = torch.arange(model.reg_max, dtype=torch.float)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=targets.device)
        else:
            # 获得图像索引
            i           = targets[:, 0]  
            _, counts   = i.unique(return_counts=True)
            out         = torch.zeros(batch_size, counts.max(), 5, device=targets.device)
            # 对batch进行循环，然后赋值
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            # 缩放到原图大小。
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            # batch, anchors, channels
            b, a, c     = pred_dist.shape  
            # DFL的解码
            pred_dist   = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.to(pred_dist.device).type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        # 然后解码获得预测框
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        # 获得使用的device
        device  = preds[1].device
        # box, cls, dfl三部分的损失
        loss    = torch.zeros(3, device=device)  
        # 获得特征，并进行划分
        feats   = preds[2] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max * 4, self.nc), 1)

        # bs, num_classes + self.reg_max * 4 , 8400 =>  cls bs, num_classes, 8400; 
        #                                               box bs, self.reg_max * 4, 8400
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        # 获得batch size与dtype
        dtype       = pred_scores.dtype
        batch_size  = pred_scores.shape[0]
        # 获得输入图片大小
        imgsz       = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * self.stride[0]  
        # 获得anchors点和步长对应的tensor
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # 把一个batch中的东西弄一个矩阵
        # 0为属于第几个图片
        # 1为种类
        # 2:为框的坐标
        targets                 = torch.cat((batch[:, 0].view(-1, 1), batch[:, 1].view(-1, 1), batch[:, 2:]), 1)
        # 先进行初步的处理，对输入进来的gt进行padding，到最大数量，并把框的坐标进行缩放
        # bs, max_boxes_num, 5
        targets                 = self.preprocess(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        # bs, max_boxes_num, 5 => bs, max_boxes_num, 1 ; bs, max_boxes_num, 4
        gt_labels, gt_bboxes    = targets.split((1, 4), 2)  # cls, xyxy
        # 求哪些框是有目标的，哪些是填充的
        # bs, max_boxes_num
        mask_gt                 = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        # 对预测结果进行解码，获得预测框
        # bs, 8400, 4
        pred_bboxes             = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # 对预测框与真实框进行分配
        # target_bboxes     bs, 8400, 4
        # target_scores     bs, 8400, 80
        # fg_mask           bs, 8400
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt
        )

        target_bboxes       /= stride_tensor
        target_scores_sum   = max(target_scores.sum(), 1)

        # 计算分类的损失
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # 计算bbox的损失
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= 7.5  # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain
        return loss.sum() # loss(box, cls, dfl) # * batch_size

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model
    
def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA:

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []
        
        os.makedirs(self.log_dir)
        
    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=0.7, label='train loss')
        plt.plot(iters, self.val_loss, 'blue', linewidth=0.7, label='val loss')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class BboxLoss(nn.Module):
    def __init__(self, reg_max=16, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # 计算IOU损失
        # weight代表损失中标签应该有的置信度，0最小，1最大
        weight      = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        # 计算预测框和真实框的重合程度
        iou         = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        # print(iou)
        # 然后1-重合程度，乘上应该有的置信度，求和后求平均。
        loss_iou    = ((1.0 - iou) * weight).sum() / target_scores_sum

        # 计算DFL损失
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        # 一个点一般不会处于anchor点上，一般是xx.xx。如果要用DFL的话，不可能直接一个cross_entropy就能拟合
        # 所以把它认为是相对于xx.xx左上角锚点与右下角锚点的距离 如果距离右下角锚点距离小，wl就小，左上角损失就小
        #                                                   如果距离左上角锚点距离小，wr就小，右下角损失就小
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)

# #获取全部类别检测的精确率
# class EvalCallback():
#     def __init__(self, net, input_shape, class_names, num_classes, val_lines, log_dir, cuda, \
#             map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, nms_iou=0.3, letterbox_image=True, MINOVERLAP=0.5, eval_flag=True, period=1):
#         super(EvalCallback, self).__init__()
        
#         self.net                = net
#         self.input_shape        = input_shape
#         self.class_names        = class_names
#         self.num_classes        = num_classes
#         self.val_lines          = val_lines
#         self.log_dir            = log_dir
#         self.cuda               = cuda
#         self.map_out_path       = map_out_path
#         self.max_boxes          = max_boxes
#         self.confidence         = confidence
#         self.nms_iou            = nms_iou
#         self.letterbox_image    = letterbox_image
#         self.MINOVERLAP         = MINOVERLAP
#         self.eval_flag          = eval_flag
#         self.period             = period
        
#         self.bbox_util          = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]))
        
#         self.maps       = [0]
#         self.epoches    = [0]
#         if self.eval_flag:
#             with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
#                 f.write(str(0))
#                 f.write("\n")

#     # def get_map_txt(self, image_id, image, class_names, map_out_path):
        
#     #     f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        
#     #     image_shape = np.array(np.shape(image)[0:2])
        
#     #     image       = cvtColor(image)
        
#     #     image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

#     #     image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

#     #     with torch.no_grad():
#     #         images = torch.from_numpy(image_data)
#     #         if self.cuda:
#     #             images = images.cuda()
#     #         #---------------------------------------------------------#
#     #         #   将图像输入网络当中进行预测！
#     #         #---------------------------------------------------------#
#     #         outputs = self.net(images)
#     #         # print(outputs)
#     #         outputs = self.bbox_util.decode_box(outputs)
#     #         # print(outputs)
#     #         #---------------------------------------------------------#
#     #         #   将预测框进行堆叠，然后进行非极大抑制
#     #         #---------------------------------------------------------#
#     #         results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
#     #                     image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
#     #         print('a')
#     #         print(results)                                        
#     #         if results[0] is None: 
#     #             return 

#     #         top_label   = np.array(results[0][:, 5], dtype = 'int32')
#     #         top_conf    = results[0][:, 4]
#     #         top_boxes   = results[0][:, :4]

#     #     top_100     = np.argsort(top_conf)[::-1][:self.max_boxes]
#     #     print('b')
#     #     print(top_100)
#     #     top_boxes   = top_boxes[top_100]
#     #     print('c')
#     #     print(top_boxes)
#     #     top_conf    = top_conf[top_100]
#     #     print('d')
#     #     print(top_conf)
#     #     top_label   = top_label[top_100]
#     #     print('e')
#     #     print(top_label)



#     #     predicted_boxes = []  # 预测框的位置信息
#     #     predicted_labels = []  # 预测框的类别
#     #     predicted_scores = []  # 预测框的置信度
        
#     #     # 记录预测结果
#     #     for i, c in list(enumerate(top_label)):
#     #         predicted_class = self.class_names[int(c)]
#     #         box = top_boxes[i]
#     #         score = top_conf[i]
        
#     #         predicted_boxes.append(box)
#     #         predicted_labels.append(predicted_class)
#     #         predicted_scores.append(score)
        
#     #     # # 筛选出置信度高于阈值的预测结果
#     #     # filtered_predicted_boxes = []
#     #     # filtered_predicted_labels = []
#     #     # filtered_predicted_scores = []
#     #     # confidence_threshold=0.6
        
#     #     # for box, label, score in zip(predicted_boxes, predicted_labels, predicted_scores):
#     #     #     if score > confidence_threshold:  # 根据置信度阈值筛选
#     #     #         filtered_predicted_boxes.append(box)
#     #     #         filtered_predicted_labels.append(label)
#     #     #         filtered_predicted_scores.append(score)
        
#     #     # 计算准确率和召回率
#     #     # 对于每个预测框，判断是否与真实目标框重叠，并计算匹配数量
#     #     matched_count = 0
#     #     for box, label in zip(predicted_boxes, predicted_labels):
#     #         # TODO: 在此处编写匹配逻辑，判断预测框是否与真实目标框重叠，计算匹配数量
#     #         for ground_truth_box, ground_truth_label in zip(ground_truth_boxes, ground_truth_labels):
#     #             # iou = calculate_iou(predicted_box, ground_truth_box)
#     #             iou=bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=True, eps=1e-7)
#     #             if iou >confidence_threshold:  # 判断 IoU 是否大于阈值
#     #                 # is_matched = True
#     #                 matched_count += 1
#     #                 break  # 如果已经匹配成功，则跳出内层循环
#     #     # 计算准确率和召回率
#     #     if len(predicted_boxes)==0:
#     #         precision=0
#     #     else:
#     #         precision = matched_count / len(predicted_boxes)
            
#     #     # if total_ground_truth_boxes_count==0:
#     #     #     recall=0
#     #     # else:
#     #     #     recall = matched_count / total_ground_truth_boxes_count





        
#     #     # precision = matched_count / len(filtered_predicted_boxes)
#     #     # recall = matched_count / total_ground_truth_boxes_count
        
#     #     print(precision)


        
#     #     for i, c in list(enumerate(top_label)):
#     #         predicted_class = self.class_names[int(c)]
#     #         box             = top_boxes[i]
#     #         score           = str(top_conf[i])

#     #         top, left, bottom, right = box
#     #         if predicted_class not in class_names:
#     #             continue
#     #         # print(predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom)))
#     #         f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

#     #     f.close()
#     #     return 
    
#     def on_epoch_end(self, epoch, model_eval):
#         if epoch % self.period == 0 and self.eval_flag:
#             self.net = model_eval
#             # print(os.path.exists(os.path.join(self.map_out_path, "ground-truth")))
#             if not os.path.exists(self.map_out_path):
#                 os.makedirs(self.map_out_path)
#             if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
#                 os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
#             if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
#                 os.makedirs(os.path.join(self.map_out_path, "detection-results"))

#             # print(os.path.exists(os.path.join(self.map_out_path, "ground-truth")))
#             #准确识别的框的数量（tp）
#             matched_count = 0
#             #tp+fp
#             all=0
#             print("Get map.")
#             for annotation_line in tqdm(self.val_lines):
#                 #超过iou

#                 line        = annotation_line.split()
#                 image_id    = os.path.basename(line[0]).split('.')[0]
               
                
#                 #------------------------------#
#                 #   读取图像并转换成RGB图像
#                 #------------------------------#
#                 image       = Image.open(line[0])
#                 #------------------------------#
#                 #   获得预测框
#                 #------------------------------#
#                 gt_boxes    = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
#                 #------------------------------#
#                 #   获得预测txt
#                 #------------------------------#
#                 # self.get_map_txt(image_id, image, self.class_names, self.map_out_path)


#                 image_shape = np.array(np.shape(image)[0:2])
                
#                 image       = cvtColor(image)
                
#                 image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        
#                 image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
                
#                 flag=True
#                 if flag:
#                     with torch.no_grad():
#                         images = torch.from_numpy(image_data)
#                         if self.cuda:
#                             images = images.cuda()
#                         #---------------------------------------------------------#
#                         #   将图像输入网络当中进行预测！
#                         #---------------------------------------------------------#
#                         outputs = self.net(images)
#                         # print(outputs)
#                         outputs = self.bbox_util.decode_box(outputs)
#                         # print(outputs)
#                         #---------------------------------------------------------#
#                         #   将预测框进行堆叠，然后进行非极大抑制
#                         #---------------------------------------------------------#
#                         results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
#                                     image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
#                         print('a')
#                         print(results)                                        
#                         if results[0] is None: 
#                             flag=False
#                             break
            
#                         top_label   = np.array(results[0][:, 5], dtype = 'int32')
#                         top_conf    = results[0][:, 4]
#                         top_boxes   = results[0][:, :4]
            
#                     top_100     = np.argsort(top_conf)[::-1][:self.max_boxes]
#                     print('b')
#                     print(top_100)
#                     top_boxes   = top_boxes[top_100]
#                     print('c')
#                     print(top_boxes)
#                     top_conf    = top_conf[top_100]
#                     print('d')
#                     print(top_conf)
#                     top_label   = top_label[top_100]
#                     print('e')
#                     print(top_label)
            
            
            
#                     predicted_boxes = []  # 预测框的位置信息
#                     predicted_labels = []  # 预测框的类别
#                     predicted_scores = []  # 预测框的置信度
#                     confidence_threshold=0.5

                    
                    
#                     # 记录预测结果
#                     for i, c in list(enumerate(top_label)):
#                         predicted_class = self.class_names[int(c)]
#                         box = top_boxes[i]
#                         score = top_conf[i]
                    
#                         predicted_boxes.append(box)
#                         predicted_labels.append(predicted_class)
#                         predicted_scores.append(score)
    
#                     all+=len(predicted_boxes)
                    
#                     for box in predicted_boxes:
#                         # TODO: 在此处编写匹配逻辑，判断预测框是否与真实目标框重叠，计算匹配数量
#                         for ground_truth_box in gt_boxes:
#                             # iou = calculate_iou(predicted_box, ground_truth_box)
#                             print('g')
#                             # print(ground_truth_box)
#                             iou=bbox_iou(box, ground_truth_box[:-1], xywh=False, GIoU=False, DIoU=False, CIoU=True, eps=1e-7)
#                             print(iou)
#                             if iou >confidence_threshold:  # 判断 IoU 是否大于阈值
#                                 # is_matched = True
#                                 matched_count += 1
#                                 break  # 如果已经匹配成功，则跳出内层循环
#                     # 计算准确率和召回率
#                     # if len(predicted_boxes)==0:
#                     #     precision=0
#                     # else:
#                     #     precision = matched_count / len(predicted_boxes)
#                     # precision = matched_count / len(predicted_boxes)
#                     # print('f')
#                     # print(precision)

#             if all!=0:
#                 # print('f')
#                 precision = matched_count / all
#             else:
#                 precision=0
#             print('f')
#             # precision = matched_count / all
#             print(precision)

                
                
#                 #------------------------------#
#                 #   获得真实框txt
#             #     #------------------------------#
#             #     with open(os.path.join(self.map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
#             #         for box in gt_boxes:
#             #             left, top, right, bottom, obj = box
#             #             obj_name = self.class_names[obj]
#             #             print('g')
#             #             print(obj_name, left, top, right, bottom)
#             #             new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
                        
#             # print("Calculate Map.")

#             # temp_map = get_map(self.MINOVERLAP, False, path = self.map_out_path)
#             # print('aaaaa')
#             # # print(temp_map)
#             # self.maps.append(temp_map)
#             # self.epoches.append(epoch)

#             # with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
#             #     f.write(str(temp_map))
#             #     f.write("\n")
            
#             # plt.figure()
#             # plt.plot(self.epoches, self.maps, 'red', linewidth = 2, label='train map')

#             # plt.grid(True)
#             # plt.xlabel('Epoch')
#             # plt.ylabel('Map %s'%str(self.MINOVERLAP))
#             # plt.title('A Map Curve')
#             # plt.legend(loc="upper right")

#             # plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
#             # plt.cla()
#             # plt.close("all")

#             print("Get map done.")
#             shutil.rmtree(self.map_out_path)

