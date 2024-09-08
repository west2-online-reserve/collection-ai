import torch
from torch import nn


def sequence_mask(X, valid_len, value=0):
    """对序列中的项进行掩码操作"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带掩码的softmax交叉熵损失函数"""
    def forward(self, pred, label, valid_length): # type: ignore
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_length)
        self.reduction = 'none'
        output = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (output * weights).mean(dim=1)
        return weighted_loss