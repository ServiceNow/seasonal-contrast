import torch
from torch import nn
from pytorch_lightning import Callback
from pl_bolts.models.self_supervised.evaluator import SSLEvaluator
from sklearn.metrics import average_precision_score

from datasets.bigearthnet_datamodule import BigearthnetDataModule


class SSLOnlineEvaluator(Callback):

    def __init__(self, data_dir, z_dim, max_epochs=10, check_val_every_n_epoch=1, batch_size=1024, num_workers=32):
        self.z_dim = z_dim
        self.max_epochs = max_epochs
        self.check_val_every_n_epoch = check_val_every_n_epoch

        self.datamodule = BigearthnetDataModule(
            data_dir=data_dir,
            train_frac=0.01,
            val_frac=0.01,
            lmdb=True,
            batch_size=batch_size,
            num_workers=num_workers
        )
        self.datamodule.setup()

        self.criterion = nn.MultiLabelSoftMarginLoss()
        self.metric = lambda output, target: average_precision_score(target, output, average='micro') * 100.0

    def on_pretrain_routine_start(self, trainer, pl_module):
        self.classifier = SSLEvaluator(
            n_input=self.z_dim,
            n_classes=self.datamodule.num_classes,
            n_hidden=None
        ).to(pl_module.device)

        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-3)

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.check_val_every_n_epoch != 0:
            return

        encoder = pl_module.encoder_q

        self.classifier.train()
        for _ in range(self.max_epochs):
            for inputs, targets in self.datamodule.train_dataloader():
                inputs = inputs.to(pl_module.device)
                targets = targets.to(pl_module.device)

                with torch.no_grad():
                    representations = encoder(inputs)
                representations = representations.detach()

                logits = self.classifier(representations)
                loss = self.criterion(logits, targets)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.classifier.eval()
        accuracies = []
        for inputs, targets in self.datamodule.val_dataloader():
            inputs = inputs.to(pl_module.device)

            with torch.no_grad():
                representations = encoder(inputs)
            representations = representations.detach()

            logits = self.classifier(representations)
            preds = torch.sigmoid(logits).detach().cpu()
            acc = self.metric(preds, targets)
            accuracies.append(acc)
        acc = torch.mean(torch.tensor(accuracies))

        metrics = {'online_val_acc': acc}
        trainer.logger_connector.log_metrics(metrics, {})
        trainer.logger_connector.add_progress_bar_metrics(metrics)
