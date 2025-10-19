import torch
import torch.nn as nn

class FineTunedTransformerClassifier(nn.Module):
    def __init__(self, pre_trained_model, num_classes=2):
        super().__init__()
        self.embedding = pre_trained_model.embedding
        self.transformer_blocks = pre_trained_model.transformer_blocks

        # Add a classification layer on top of the transformer blocks
        self.classification_layer = nn.Linear(pre_trained_model.hidden, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, segment_info):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x, segment_info)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        classification_output = self.classification_layer(x[:, 0])

        classification_probs = self.softmax(classification_output)

        return classification_probs

