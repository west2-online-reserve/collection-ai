import torch
from torch.utils.data import Dataset

class MyFineTuneDataset(Dataset):
    def __init__(self, data, vocab, max_seq_length=128):
        self.data = data
        self.vocab = vocab
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        sentence1 = item["sentence1"]
        sentence2 = item["sentence2"]
        label = int(item["label"])

        # Tokenize and convert to sequences
        seq1 = self.vocab.to_seq(sentence1, seq_len=self.max_seq_length - 3)  # -3 to account for [CLS], [SEP], [SEP]
        seq2 = self.vocab.to_seq(sentence2, seq_len=self.max_seq_length - len(seq1) - 3)

        # Combine tokens: [CLS] + seq1 + [SEP] + seq2 + [SEP]
        input_ids = [self.vocab.cls_index] + seq1 + [self.vocab.sep_index] + seq2 + [self.vocab.sep_index]

        # Pad the sequence if needed
        input_ids += [self.vocab.pad_index] * (self.max_seq_length - len(input_ids))

        # Convert sequence to a tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Create attention mask
        attention_mask = (input_ids != self.vocab.pad_index).float()

        # Create segment information
        segment_info = torch.tensor([0] * (len(seq1) + 2) + [1] * (len(seq2) + 1), dtype=torch.long)

        # Label tensor
        label_tensor = torch.tensor(label, dtype=torch.long)

        return input_ids, attention_mask, segment_info, label_tensor
