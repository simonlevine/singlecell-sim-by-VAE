import warnings
import json
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as metrics
from pytorch_lightning.loggers import WandbLogger
from loguru import logger
from pipeline.datalib import load_single_cell_data, load_combined_single_cell_data
from pipeline.helpers.paths import CLASSIFER_W_SIMULATED_DATA_METRICS, CLASSIFER_WOUT_SIMULATED_DATA_METRICS
from pipeline.helpers.params import params


def main():
    for metrics_fp, datamodule_ in [
        (CLASSIFER_W_SIMULATED_DATA_METRICS, load_combined_single_cell_data),
        (CLASSIFER_WOUT_SIMULATED_DATA_METRICS, load_single_cell_data),
    ]:
        logger.info("loading data (takes about a minute)")
        datamodule = datamodule_()
        model = SingleCellClassifier()
        trainer = pl.Trainer(
            gpus=torch.cuda.device_count(),
            logger=WandbLogger(project="02718-vae"),
            **params.classifier.training_opts
        )
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=UserWarning)  # AnnData  is single threaded
            trainer.fit(model, datamodule)
            trainer.test(model, test_dataloaders=datamodule.test_dataloader())
        with open(metrics_fp, "w") as f:
            m = {k: v.item() for k, v in trainer.callback_metrics.items()}
            json.dump(m, f)


class SingleCellClassifier(pl.LightningModule):
    
    def __init__(self, input_size = 26361, learning_rate=1e-3, hidden_size=2048):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_classes = 3    # healthy, sick or sick+ventilated
        fc_block = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1))
        self.fc1 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            *[deepcopy(fc_block) for _ in range(params.classifier.n_hidden_layers)])
        self.output = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self,x):
        x = self.fc1(x)
        x = self.output(x)
        return F.log_softmax(x, dim=1)

    def _step(self, gene_expression, ventilator_status):
        logits = self.forward(gene_expression)
        y_pred = torch.argmax(logits, dim=1)
        loss = F.nll_loss(logits, ventilator_status)
        return y_pred, loss

    def training_step(self, batch, batch_idx):
        gene_expression, _, ventilator_status = batch
        _, loss = self._step(gene_expression, ventilator_status)
        return loss

    def validation_step(self, batch, batch_idx):
        gene_expression, _, ventilator_status = batch
        y_pred, loss = self._step(gene_expression, ventilator_status)
        self.log("val_loss", loss)
        return {"val_loss": loss,
                "y_true": ventilator_status,
                "y_pred": y_pred}

    def test_step(self, batch, batch_idx):
        gene_expression, _, ventilator_status = batch
        y_pred, loss = self._step(gene_expression, ventilator_status)
        self.log("test_loss", loss)
        return {"test_loss": loss,
                "y_true": ventilator_status,
                "y_pred": y_pred}

    def _epoch_end(self, stage, steps):
        y_true = torch.vstack([x["y_true"] for x in steps]).reshape(-1,1)
        y_pred = torch.vstack([x["y_pred"] for x in steps]).reshape(-1,1)
        return {
            f"{stage}_acc": metrics.accuracy(y_pred, y_true),
            f"{stage}_f1": metrics.f1(y_pred, y_true, num_classes=self.num_classes),
            f"{stage}_recall": metrics.recall(y_pred, y_true),
            f"{stage}_precision": metrics.precision(y_pred, y_true),
        }

    def validation_epoch_end(self, validation_step_outputs):
        return self._epoch_end("validation", validation_step_outputs)

    def test_epoch_end(self, test_step_outputs):
        return self._epoch_end("test", test_step_outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":
    main()