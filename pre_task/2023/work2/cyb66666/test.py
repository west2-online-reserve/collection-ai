import torch
import pickle
from train import test_imdb_dataloader, train_imdb_dataloader

net = pickle.load(open('./modul.pkl', 'rb'))  # 运行train.py后才有model.pkl
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for (x, y) in test_imdb_dataloader:
        x = torch.as_tensor(x).to(device)
        y = torch.as_tensor(y).to(device)
        outputs = net.forward(x)
        for i, output in enumerate(outputs):
            if y[i][torch.argmax(output)] == 1:
                correct += 1
            total += 1
    print(correct / total)
