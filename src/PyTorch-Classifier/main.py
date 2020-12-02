import pytorch_lightning as pl
import os

import torch
from torch import nn
from torch.nn import functional as F
# from torch.utils.data import DataLoader, 
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as metrics

from pipeline.datalib import load_single_cell_data

def main():

    dm = load_single_cell_data()
    model = SingleCellClassifier()
    trainer = pl.Trainer(gpus = torch.cuda.device_count(), fast_dev_run = True)
    trainer.fit(model = model, datamodule= dm)


class SingleCellClassifier(pl.LightningModule):
    
    def __init__(self, input_size = 26361, learning_rate=1e-3):

        super().__init__()

        self.hidden_size = 2048
        self.learning_rate = learning_rate

        self.num_classes = 3

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_size, 4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.1))

        self.output = nn.Linear(1024, self.num_classes)

    def forward(self,x): #,skip_conn=False):
        # if skip_conn:
        #     residual = x
        #     x = self.fc1(x)
        #     x = self.fc2(x)
        #     x = self.fc3(x+residual)
        #     x = self.fc4(x+residual)
        # else:
        
        x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        # x = self.fc4(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = metrics.accuracy(preds, y)
        f1 = metrics.f1_score(preds,y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda ep: 1 / (1 + 0.05 * ep), last_epoch=-1, verbose=False)
        return [optimizer], [scheduler]