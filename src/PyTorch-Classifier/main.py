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
from pytorch_lightning import loggers as pl_loggers

from pipeline.datalib import load_single_cell_data, load_combined_single_cell_data

def main():

    dm = load_single_cell_data()
    model = SingleCellClassifier()
    tb_logger = pl_loggers.TensorBoardLogger('baseline_logs/')

    trainer = pl.Trainer(gpus = torch.cuda.device_count(), fast_dev_run = True, logger=tb_logger)
    trainer.fit(model = model, datamodule= dm)
    trainer.test(model, test_dataloaders=dm.test_dataloader())


    dm = load_combined_single_cell_data()
    model = SingleCellClassifier()
    tb_logger = pl_loggers.TensorBoardLogger('simulated_logs/')

    trainer = pl.Trainer(gpus = torch.cuda.device_count(), fast_dev_run = True, logger=tb_logger)
    trainer.fit(model = model, datamodule= dm)
    trainer.test(model, test_dataloaders=dm.test_dataloader())



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

    def forward(self,x):
        x = self.fc1(x)
        x = self.output(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = metrics.accuracy(preds, y)
        f1 = metrics.f1_score(preds,y)
        recall = metrics.recall(preds,y)
        precision=metrics.precision(preds,y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_f1',f1,prog_bar=True)
        self.log('val_recall',recall,prog_bar=True)
        self.log('val_precision',precision,prog_bar=True)
        return {'val_loss':loss, 'val_acc': acc, 'val_f1':f1,'val_recall':recall,'val_precision':precision}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        acc = metrics.accuracy(preds, y)
        f1 = metrics.f1_score(preds,y)
        recall = metrics.recall(preds,y)
        precision=metrics.precision(preds,y)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_f1',f1,prog_bar=True)
        self.log('test_recall',recall,prog_bar=True)
        self.log('test_precision',precision,prog_bar=True)
        
        return {'test_loss':loss, 'test_acc': acc, 'test_f1':f1,'test_recall':recall,'test_precision':precision}
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda ep: 1 / (1 + 0.05 * ep), last_epoch=-1, verbose=False)
        return [optimizer], [scheduler]