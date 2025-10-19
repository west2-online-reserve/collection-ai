import torch
import time

def train(model, epoch, dataloader, crit, optim, device, ep):
    model.train()
    for i, (img, target) in enumerate(dataloader):
        img = img.to(device)
        target = target.to(device)
        optim.zero_grad()
        res = model(img)
        loss, each_loss = crit(res, target)
        loss.backward()
        optim.step()
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1:>3d} / {ep:>3d}] Step [{(i + 1) :>4d} / {len(dataloader):>4d}] Loss: {loss.item() :>7f} lr: {optim.param_groups[0]["lr"]}')

def test(model, dataloader, crit, device, computemAP):
    model.eval()
    Loss = 0
    mAP = 0
    rec = 0
    for i, (img, target) in enumerate(dataloader):
        bs = img.shape[0]
        img = img.to(device)
        target = target.to(device)
        res = model(img)
        loss, each_loss = crit(res, target)
        Loss += loss.item()
        ap, rc = computemAP(res, target)
        mAP += ap * bs
        rec += rc * bs
        # print(mAP)
        if (i + 1) % 10 == 0:
            print(i, loss.item(), len(dataloader))
        #     time.sleep(1)
    Loss /= len(dataloader)
    mAP  /= len(dataloader)
    rec  /= len(dataloader)
    print(f'Test: \n Avg loss:{Loss :> 8f}, mAP:{mAP * 100 :> 4f}%, Recall:{rec * 100 :> 4f}%')