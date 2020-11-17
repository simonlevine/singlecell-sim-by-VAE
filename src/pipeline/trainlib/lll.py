from loguru import logger
from pytorch_lightning.loggers import LightningLoggerBase

class LoguruLightningLogger(LightningLoggerBase):

    def __init__(self):
        super().__init__()
        self.logger = logger

    def log_metrics(self, metrics, step):
        logger.info("{} : {}", step, metrics)

    def name(self):
        return "Jeremiah's Loguru Lightning Logger"

    def experiment(self):
        ...

    def log_hyperparams(self, params):
        ...

    def version(self):
        return '0.1'