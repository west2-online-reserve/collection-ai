import json
import torch
from tqdm import tqdm

from transformers import BertTokenizer
from torch.utils.data import Dataset


class BertGenerateDataset(Dataset):
    def __init__(
        self,
        corpus_path: str,
        model_path: str = "bert-base-chinese",
        encoding: str = "utf-8",
    ) -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.vocab_size = len(self.tokenizer.vocab)
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.data = self.load_data()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple:
        input_ids, label_ids = self.data[index]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        input_valid_len = torch.tensor(len(input_ids), dtype=torch.long)
        label_valid_len = torch.tensor(len(label_ids), dtype=torch.long)
        return input_ids, input_valid_len, label_ids,  label_valid_len

    def load_data(self):
        with open(self.corpus_path, "r", encoding=self.encoding) as f:
            data = [json.loads(line) for line in f]
            data = self.process_data(data)
        return data

    def process_data(self, data):
        processed_data = []
        for index in tqdm(range(len(data))):
            input_text = "[CLS]" + data[index]["sentence_1"] + "[SEP]"
            target_text = data[index]["sentence_2"]
            input_text_tokened = self.tokenizer.tokenize(input_text)
            input_ids = self.tokenizer.convert_tokens_to_ids(input_text_tokened)
            target_text_tokened = self.tokenizer.tokenize(target_text)
            target_ids = self.tokenizer.convert_tokens_to_ids(target_text_tokened)
            processed_data.append((input_ids, target_ids))
        return processed_data
