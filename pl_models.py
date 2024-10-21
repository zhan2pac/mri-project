import torch
from torch import optim

# from ranger21 import Ranger21
from transformers import get_cosine_schedule_with_warmup
import pytorch_lightning as pl
from attrdict import AttrDict
import math
from models import TorchModel
from losses import TorchLoss
import os
import pandas as pd
from metrics import class_metrics_multi, class_metrics_binary
from collections import OrderedDict
import numpy as np
import segmentation_models_pytorch_3d.metrics as smpm


class TrainModel(pl.LightningModule):
    def __init__(self, config, train_loader, val_loader):
        super(TrainModel, self).__init__()
        self.config = config
        self.num_classes = config["model"]["num_classes"]

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.num_training_steps = math.ceil(len(self.train_loader) / len(config["trainer"]["devices"]))
        self.model = TorchModel(config["model"])
        self.criterion = TorchLoss(config["loss"])

        self.save_hyperparameters(AttrDict(config))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log("loss/train", loss, on_step=False, on_epoch=True, rank_zero_only=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        tp, fp, fn, tn = smpm.get_stats(
            torch.argmax(y_hat, dim=1), y, mode="multiclass", num_classes=self.num_classes, ignore_index=-1
        )
        return {"loss": loss.cpu(), "tp": tp.cpu(), "fp": fp.cpu(), "fn": fn.cpu(), "tn": tn.cpu()}

    def sync_across_gpus(self, tensors):
        tensors = self.all_gather(tensors)
        return torch.cat([t for t in tensors])

    def validation_epoch_end(self, outputs):
        out_val = {}
        for key in outputs[0].keys():
            if key == "loss":
                out_val[key] = torch.stack([o[key] for o in outputs])
            else:
                out_val[key] = torch.cat([o[key] for o in outputs], dim=0)

        # for key in out_val.keys():
        #     out_val[key] = self.sync_across_gpus(out_val[key])

        loss = out_val["loss"].mean()
        self.log("loss/val", loss, prog_bar=False, rank_zero_only=True, sync_dist=False)
        # result_metrics = class_metrics_multi(out_val['output'], out_val['target'])
        result_metrics = class_metrics_multi(out_val["tp"], out_val["fp"], out_val["fn"], out_val["tn"])
        for metric, result in result_metrics.items():
            if len(result) > 1:
                for i, res in enumerate(result):
                    self.log(f"{metric}/class_{i}", res, sync_dist=False)
                self.log(f"{metric}/mean", result.mean(), sync_dist=False)
            else:
                self.log(f"{metric}/mean", np.mean(result), sync_dist=False)

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()), **self.config["optimizer_params"]
            )
        # elif self.hparams.optimizer == "ranger21":
        #     optimizer = Ranger21(
        #         filter(lambda p: p.requires_grad, self.model.parameters()),
        #         num_batches_per_epoch=self.num_training_steps,
        #         num_epochs=self.config["trainer"]["max_epochs"],
        #         **self.config['optimizer_params']
        #     )
        elif self.hparams.optimizer == "sgd":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                momentum=0.9,
                nesterov=True,
                **self.config["optimizer_params"],
            )
        else:
            raise ValueError(f"Unknown optimizer name: {self.hparams.optimizer}")

        scheduler_params = AttrDict(self.hparams.scheduler_params)
        if self.hparams.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=scheduler_params.patience,
                min_lr=scheduler_params.min_lr,
                factor=scheduler_params.factor,
                mode=scheduler_params.mode,
                verbose=scheduler_params.verbose,
            )

            lr_scheduler = {"scheduler": scheduler, "interval": "epoch", "monitor": scheduler_params.target_metric}
        elif self.hparams.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.num_training_steps * scheduler_params.warmup_epochs,
                num_training_steps=int(self.num_training_steps * self.config["trainer"]["max_epochs"]),
            )

            lr_scheduler = {"scheduler": scheduler, "interval": "step"}
        else:
            raise ValueError(f"Unknown sheduler name: {self.hparams.sheduler}")

        return [optimizer], [lr_scheduler]


class TestModel(pl.LightningModule):
    def __init__(self, config, test_loader, test_dataset):
        super(TestModel, self).__init__()
        self.config = config
        self.test_loader = test_loader
        self.test_dataset = test_dataset
        self.model = TorchModel(**config["model"])
        state_dict = torch.load(config["weights"], map_location="cpu")["state_dict"]
        self.load_state_dict(state_dict, strict=True)
        self.save_hyperparameters(AttrDict(config))
        self.softmax = torch.nn.Softmax(dim=1)

    def test_dataloader(self):
        return self.test_loader

    def test_step(self, batch, batch_idx):
        x, ids = batch
        y_hat = self.model(x)
        y_hat = [self.softmax(y) for y in y_hat]
        # y_hat = torch.argmax(y_hat, dim=1)
        return {"y_hat_1": y_hat[0], "y_hat_2": y_hat[1], "idx": ids}

    def sync_across_gpus(self, tensors):
        tensors = self.all_gather(tensors)
        return torch.cat([t for t in tensors])

    def test_epoch_end(self, outputs):
        y_hat_1 = torch.cat([o["y_hat_1"] for o in outputs], dim=0).cpu().detach().tolist()
        y_hat_2 = torch.cat([o["y_hat_2"] for o in outputs], dim=0).cpu().detach().tolist()
        ids = torch.cat([o["idx"] for o in outputs], dim=0).cpu().detach().tolist()
        data = [
            [self.test_dataset.images[idx], prediction_1, prediction_2]
            for idx, prediction_1, prediction_2 in zip(ids, y_hat_1, y_hat_2)
        ]
        df = pd.DataFrame(data, columns=["filename", *self.hparams.classnames]).drop_duplicates()
        file_path = os.path.join(self.hparams.save_path, self.hparams.test_name, "predictions.csv")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
