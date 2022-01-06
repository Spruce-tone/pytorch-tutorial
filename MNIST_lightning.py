import torch
from torch import nn
import pytorch_lightning as pl
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Int
from torch.nn import functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT

class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self):
        super(LightningMNISTClassifier, self).__init__()

        # mnist images are (1, 28, 28) (channels, width, height) 
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x: torch.Tensor):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3
        x = self.layer_3(x)

        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)

        return x
    
    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def cross_entropy_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch: torch.Tensor, batch_idx: int) -> Optional[STEP_OUTPUT]:
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log = ('val_loss', loss)