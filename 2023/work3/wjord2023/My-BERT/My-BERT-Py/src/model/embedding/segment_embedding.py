from torch import nn

# 为输入的句子中的每个词生成对应的段（segment）嵌入（embedding）
class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)