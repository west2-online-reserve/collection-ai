import re
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter


class ScheduleOptim:
    def __init__(
        self,
        optimizer,
        d_model,
        n_warmup_steps,
        gamma=None,
        scheduler=None,
        writer=None,
    ) -> None:
        self._optimizer = optimizer
        self.n_current_steps = 0
        self.init_lr = self._optimizer.param_groups[0]["lr"]
        self.gamma = gamma
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self._scheduler = scheduler or self._default_lr_scheduler()
        self.writer = writer

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step_and_update_lr(self):
        self._optimizer.step()
        self.n_current_steps += 1
        self._scheduler.step()
        if self.writer is not None:
            self.writer.add_scalar("lr", self.get_lr(), self.n_current_steps)

    def get_lr(self):
        return self._optimizer.param_groups[0]["lr"]

    def reset(self):
        self.n_current_steps = 0
        self._scheduler.last_epoch = 0
        self._optimizer.param_groups[0]["lr"] = self.init_lr

    def _default_lr_scheduler(self):
        def lr_lambda(step):
            if step == 0:
                return self.init_lr
            return (
                np.power(self.d_model, -0.5)
                * np.min(
                    [
                        np.power(step, -0.5),
                        step * np.power(self.n_warmup_steps, -1.5),
                    ]
                )
            )

        return LambdaLR(self._optimizer, lr_lambda=lr_lambda)
