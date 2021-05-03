from argparse import ArgumentParser
from itertools import chain

import torch
from torch import nn, optim
from pytorch_lightning import LightningModule
from pl_bolts.models.self_supervised import SSLEvaluator
from sklearn.metrics import average_precision_score


class SSLFineTuner(LightningModule):

    def __init__(self, backbone, in_features, num_classes, hidden_dim=1024, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)

        self.backbone = backbone
        self.ft_network = SSLEvaluator(
            n_input=in_features,
            n_classes=num_classes,
            p=0.2,
            n_hidden=hidden_dim
        )
        self.criterion = nn.MultiLabelSoftMarginLoss()

    def forward(self, x):
        with torch.set_grad_enabled(not self.hparams.freeze_backbone):
            feats = self.backbone(x)
        logits = self.ft_network(feats)
        return logits

    def on_train_epoch_start(self) -> None:
        self.backbone.eval()

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log_dict({'acc/train': acc, 'loss/train': loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log_dict({'acc/val': acc, 'loss/val': loss}, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        self.log_dict({'acc/test': acc, 'loss/test': loss})
        return loss

    def shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = average_precision_score(y.cpu(), torch.sigmoid(logits).detach().cpu(), average='micro') * 100.0
        return loss, acc

    def configure_optimizers(self):
        params = self.ft_network.parameters()
        if not self.hparams.freeze_backbone:
            params = chain(self.backbone.parameters(), params)
        optimizer = optim.Adam(params, lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.milestones)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers', type=int, default=32)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--learning_rate', type=float, default=1e-5)
        parser.add_argument('--freeze_backbone', action='store_true')
        parser.add_argument('--milestones', type=int, nargs='*', default=[60, 80])
        return parser
