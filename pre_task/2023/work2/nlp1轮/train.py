import torch
import time
from utils import binary_acc

def train(rnn, iterator, optimizer, criteon):
    since = time.time()
    epoch_acc = 0
    epoch_loss = 0

    rnn.train()
    for batch in iterator:
        # [seq, b] => [b, 1] => [b]
        pred = rnn(batch.text).squeeze(1)
        #
        loss = criteon(pred, batch.label)
        acc = binary_acc(pred, batch.label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        time_elapsed = time.time() - since
        epoch_acc += acc.item()
        epoch_loss += loss.item()

    acc1 = epoch_acc / len(iterator)
    loss1 = epoch_loss / len(iterator)
    print(
        "Time elapsed {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print(f'acc:{acc1} ,loss:{loss1}')
    return acc1, loss1