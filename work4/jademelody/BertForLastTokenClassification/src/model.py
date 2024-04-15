from transformers import BertModel, BertConfig, BertTokenizer
from torch import nn
import torch

class BertForLastTokenClassification(nn.Module):
    def __init__(self, model_name: str = "bert-base-chinese", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.config = BertConfig.from_pretrained(model_name)
        self.config.num_labels = len(self.tokenizer.vocab)
        self.cls = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, input_ids) -> torch.Tensor:
        atten_mask = input_ids.ne(0)
        output = self.bert(input_ids, attention_mask=atten_mask)
        last_hidden_state = output.last_hidden_state
        last_token = last_hidden_state[:, -1, :]
        logits = self.cls(last_token)
        return self.softmax(logits)