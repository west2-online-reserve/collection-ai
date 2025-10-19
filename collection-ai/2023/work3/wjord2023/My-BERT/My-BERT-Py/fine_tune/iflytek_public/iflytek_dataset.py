import json

class IflytekDataset:
    def __init__(self, data_path, vocab, max_len):
        self.data_path = data_path
        self.vocab = vocab
        self.max_len = max_len
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]