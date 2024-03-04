import torch
from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset

import time
from torch import optim
from bertB.src.modeling import BertForMaskedLM
from bertB.src.modeling import BertConfig
from bertB.src.utils import TextDataset,mask_tokens


token = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='/tmp/pycharm_project_392/bert',
    cache_dir=None,
    force_download=False,
)

bertConfig=BertConfig(vocab_size=token.vocab_size)
bertlm=BertForMaskedLM(bertConfig)
# checkpoint = torch.load("/tmp/pycharm_project_392/bestBert17.pt")
# bertlm.load_state_dict(checkpoint)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder1 = torch.nn.Linear(token.vocab_size, 768, bias=False)
        self.decoder2 = torch.nn.Linear(768, token.vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(token.vocab_size))
        self.bertlm=bertlm
        self.decoder2.bias = self.bias


    def forward(self, input_ids, attention_mask, token_type_ids):

        out = self.bertlm(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        out=self.decoder1(out)
        out=self.decoder2(out)

        return out[:,15]