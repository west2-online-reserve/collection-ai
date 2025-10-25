import torch
from torch import nn
from torch.nn import functional as F


class Classifier(nn.Module):
    """基础分类器接口"""

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot("loss", self.loss(Y_hat, batch[-1]), train=False)
        self.plot("acc", self.accuracy(Y_hat, batch[-1]), train=False)

    def accuracy(self, Y_hat, Y, averaged=True):
        """准确率"""
        Y_hat = Y_hat.reshape(-1, Y_hat.shape[-1])
        preds = Y_hat.argmax(axis=1).type(Y.dtype)
        compare = (preds == Y.reshape(-1)).type(Y.dtype)
        return compare.mean() if averaged else compare

    def loss(self, Y_hat, Y, averaged=True):  # type: ignore
        Y_hat = Y_hat.reshape(-1, Y_hat.shape[-1])
        Y = Y.reshape(-1)
        return F.cross_entropy(Y_hat, Y, reduction="mean" if averaged else "none")

    def layer_summary(self, X_shape):
        """打印网络结构"""
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, "output shape:\t", X.shape)


class EncoderDecoder(Classifier):
    """编码器解码器基类"""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):  # type: ignore
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        return self.decoder(dec_X, dec_state)[0]

    def predict_step(self, batch, device, num_steps, save_attention_weights=False):
        batch = [X.to(device) for X in batch]
        src, tgt, src_valid_len, _ = batch
        enc_all_outputs = self.encoder(src, src_valid_len)
        dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
        outputs, attention_weights = [
            (tgt[:, 0]).unsqueeze(1),
        ], []
        for _ in range(num_steps):
            Y, dec_state = self.decoder(outputs[-1], dec_state)
            outputs.append(Y.argmax(dim=2))
            if save_attention_weights:
                attention_weights.append(self.decoder.attention_weights)
            return torch.concat(outputs[1:], dim=1), attention_weights
