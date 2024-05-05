import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.mAP import mAP
from tools import Match

matcher = Match(7, 7)

def train(model, epoch, dataloader, crit, optims, device, ep, scheduler):
    model.train()
    ls = []
    se = []
    for i, (img, lable, bbox) in enumerate(dataloader):
        # fig, ax = plt.subplots()
        # ax.imshow(img[0].permute(1, 2, 0))
        for i in range(bbox[0].shape[0]):
            if(lable[0][i] == -1):
                continue
        #     print(bbox[0][i])
        #     box = patches.Rectangle((bbox[0][i][0] * 224, bbox[0][i][1] * 224), bbox[0][i][2] * 224, bbox[0][i][3] * 224, linewidth = 2, edgecolor = 'r', facecolor = 'none')
        #     ax.add_patch(box)
        #     plt.text(bbox[0][i][0].detach().numpy() * 224, bbox[0][i][1].detach().numpy() * 224, f"{lable[0][i]}", color= "red")
        # plt.axis('off')
        # plt.show()
        img = img.to(device)
        lable, bbox = matcher(lable, bbox)
        lable = lable.to(device)
        bbox = bbox.to(device)
        optims.zero_grad()
        res = model(img).view(img.shape[0], 7, 7, -1)
        Loss = crit(res, lable, bbox, device)
        Loss.backward()
        optims.step()
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1:>3d} / {ep:>3d}] Step [{(i + 1) :>4d} / {len(dataloader):>4d}] Loss: {Loss :>7f} lr: {optims.param_groups[0]["lr"]}')
            ls.append(Loss.item())
            se.append(epoch + float(i) / float(len(dataloader)))
            # scheduler.step()
    return ls, se
        
def Gb(x = 7, y = 7):
        res_x, res_y = torch.meshgrid([torch.arange(x), torch.arange(y)])
 
        res = torch.stack((res_x, res_y), dim = -1).float()

        return res

def test(model, dataloader, crit, device, conf_threshold = 0.1):
    model.eval()
    Loss, mAP_ = 0, 0
    targets = [[]for _ in range(20)]
    cnt = torch.zeros(20)
    Gdb = Gb().unsqueeze(0).unsqueeze(3).to(device)
    mx = 0
    mxb = None
    with torch.no_grad():
        for img, label, bbox in dataloader:
            label, bbox = matcher(label, bbox)
            img, label, bbox = img.to(device), label.to(device), bbox.to(device)
            res = model(img).view(img.shape[0], 7, 7, -1)
            losss = crit(res, label, bbox, device)
            Loss += losss.item()
            res_boxes = res[..., :10]
            res_cls = torch.softmax(res[..., 10:], dim = -1)
            res_boxes = res_boxes.reshape(res_boxes.shape[0], 7, 7, 2, 5)
            res_boxes[..., 0:2] = (torch.sigmoid(res_boxes[..., 0:2]) + Gdb) / 7.
            res_boxes[..., 2:4] = torch.sigmoid(res_boxes[..., 2:4])
            res_conf = res_boxes[..., 4]
            max_conf, max_conf_id = torch.max(res_conf, dim = 3)
            max_cls, max_cls_id = torch.max(res_cls, dim = 3)
            best_box = torch.gather(res_boxes, dim = 3, index = max_conf_id.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 5)).squeeze().to(device)
            conf = max_conf * max_cls
            # print(type((conf > conf_threshold).float()), type(conf > conf_threshold))
            cls = ((max_cls_id + 1) * (conf > conf_threshold).float()) - 1
            cls_true = (cls == label).float().reshape(-1, 1)
            cls = cls.reshape(-1, 1)
            best_box = best_box.reshape(-1, 5)
            label = label.reshape(-1, 1)
            bbox = bbox.reshape(-1, 4)
            # print(torch.max(conf))
            for i in range(bbox.shape[0]):
                if(label[i][0] != -1):
                    cnt[label[i][0]] += 1
                if(cls[i][0] != -1):
                    # print(cls[i][0])
                    if(cls_true[i][0] == 1.0):
                        targets[int(cls[i][0])].append(torch.cat((best_box[i], bbox[i], label[i])))
                    else:
                        targets[int(cls[i][0])].append(torch.cat((best_box[i], bbox[i], torch.tensor([-1]).to(device))))
    
    # targets = torch.tensor(targets).to(device)
    # print(targets.shape)
    mAP_, recall = mAP(targets, cnt, device)
    Loss /= len(dataloader)
    print(f'Test Error: \n mAP : {(mAP_ * 100):> 0.1f} % ,Avg loss:{Loss :> 8f}, Recall: {(recall * 100):>0.01f} % \n')
    return mAP_, recall, Loss
