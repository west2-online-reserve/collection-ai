from dataclasses import dataclass

import torch
from torch.nn.modules import Module


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 10
    log_freq: int = 20
    with_cuda: bool = True
    with_scheduler: bool = True
    count_correct: bool = False

    hidden: int = 512
    n_warmup_steps: int = 400
    gamma = 1.0
