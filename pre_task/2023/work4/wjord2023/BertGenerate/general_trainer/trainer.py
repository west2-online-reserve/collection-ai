from typing import Callable

import torch
from torch import nn
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter  
from tqdm import tqdm

from .config import TrainerConfig
from .optim_schedule import ScheduleOptim


class GeneralTrainer:
    def __init__(
        self,
        model,
        criterion,
        train_loader,
        optimizer=None,
        valid_loader=None,
        test_loader=None,
        scheduler=None,
        cal_loss=None,
        data_process=None,
        count_acc=None,
        config=None,
    ) -> None:
        if config is None:
            config = TrainerConfig()
        cuda_condition = config.with_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda_condition else "cpu")
        self.model = model.to(self.device)

        if config.with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUs for training" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        if scheduler is None and optimizer is not None:
            self.optim = optimizer
        if scheduler and optimizer is None:
            raise ValueError
        self.with_scheduler = config.with_scheduler
        self.writer = SummaryWriter()
        self.writer.flush()
        if config.with_scheduler:
            if scheduler is None:
                self.optim_scheduler = ScheduleOptim(
                    self.optim,
                    d_model=config.hidden,
                    n_warmup_steps=config.n_warmup_steps,
                    gamma=config.gamma,
                    scheduler=scheduler,
                    writer=self.writer,
                )
            else:
                self.optim_scheduler = scheduler

        self.cal_loss = cal_loss
        self.data_process = data_process
        self.count_acc = count_acc
        self.criterion = criterion
        self.epochs = config.epochs
        self.log_freq = config.log_freq
        self.count_correct = config.count_correct

        print(
            "Total number of parameters: ", sum([p.numel() for p in model.parameters()])
        )
        print("Using device: ", self.device)

    def train(self, epoch):
        self.iteration(epoch, self.train_loader, "train")

    def test(self, epoch):
        self.iteration(epoch, self.test_loader, "test")

    def valid(self, epoch):
        self.iteration(epoch, self.valid_loader, "valid")

    def iteration(self, epoch, data_loader, mode):
        data_itr = tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
        )

        avg_loss = 0.0
        running_loss = 0.0
        if self.count_correct:
            total_correct = 0
            total_element = 0

        for batch_idx, data in data_itr:
            if self.cal_loss is None:
                data = self._process_data(data)
                loss = self._calculate_loss(data)
            else:
                loss = self.cal_loss(data, self.model)  # type: ignore

            if mode == "train":
                if self.with_scheduler:
                    self.optim_scheduler.zero_grad()
                    loss.backward()
                    self.optim_scheduler.step_and_update_lr()
                else:
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

            avg_loss += loss.item()
            running_loss += loss.item()

            if batch_idx % self.log_freq == self.log_freq - 1:
                self.writer.add_scalar(
                    f"Loss/{mode}",
                    running_loss / self.log_freq,
                    epoch * len(data_loader) + batch_idx,
                )
                running_loss = 0.0
                post_fix = {
                    "epoch": epoch,
                    "iter": batch_idx,
                    "avg_loss": avg_loss / (batch_idx + 1),
                    "loss": loss.item(),
                }
                if self.count_correct:
                    if self.count_acc is None:
                        total_correct += self._count_total_correct(data)  # type: ignore
                        total_element += self._count_total_element(data)  # type: ignore
                    else:
                        total_correct, total_element = self.count_acc(data)
                    post_fix["accuracy"] = total_correct / total_element * 100
                    self.writer.add_scalar(
                        f"Acc/{mode}",
                        total_correct / total_element * 100,
                        epoch * len(data_loader) + batch_idx,
                    )
                data_itr.set_postfix(post_fix)
        print("EP_%s:%d avg_loss:%.4f" % (mode, epoch, avg_loss / len(data_itr)))

    def _process_data(self, data):
        if isinstance(data, tuple):
            try:
                inputs, targets = data
            except ValueError:
                raise ValueError("Data must be tuple of (inputs, targets)")
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            return {"inputs": inputs, "targets": targets}
        elif isinstance(data, dict):
            for key, value in data.items():
                data[key] = value.to(self.device)
            return data
        elif isinstance(data, list):
            if len(data) != 2:
                raise ValueError("Data must be list of [inputs, targets]")
            inputs = data[0].to(self.device)
            targets = data[1].to(self.device)
            return {"inputs": inputs, "targets": targets}
        else:
            raise ValueError(
                "Data must be tuple dict or list, not {}".format(type(data))
            )

    def _calculate_loss(self, data):
        inputs = data["inputs"]
        targets = data["targets"]
        outputs = self.model(inputs, targets)
        loss = self.criterion(outputs, targets)
        return loss

    def _count_total_correct(self, data):
        output = self.model(data["inputs"])
        return (output.argmax(dim=1) == data["targets"]).sum().item()

    def _count_total_element(self, data):
        return data["targets"].shape[0]

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def training(self):
        for epoch in range(self.epochs):
            self.train(epoch)
            if self.valid_loader:
                self.valid(epoch)
        if self.test_loader:
            self.test(self.epochs)
        self.writer.close()
