#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import SingleDeviceStrategy, DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from datasets import get_dataloader
from pl_models import TrainModel
import shutil
from loguru import logger

from utils import ROOT_PATH, read_yaml
from torchinfo import summary

torch.cuda.empty_cache()


@rank_zero_only
def check_dir(dirname):
    if not os.path.exists(dirname):
        return

    print(f"Save directory - {dirname} exists")
    print("Ignore: Yes[y], No[n]")
    ans = input().lower()
    if ans == "y":
        shutil.rmtree(dirname)
        return

    raise ValueError("Tried to log experiment into existing directory")


def parse_args(args):
    parser = argparse.ArgumentParser(description="Template for training networks with pytorch lightning.")

    parser.add_argument(
        "config",
        help="path to yaml config file",
        default="/home/arseniy.zemerov/Research/arseniy.zemerov/NaiveUniform/configs/train.yaml",
    )
    return parser.parse_args(args)


@logger.catch
def train(config):
    config["save_path"] = os.path.join(config["exp_path"], config["project"], config["exp_name"])

    check_dir(config["save_path"])
    os.makedirs(config["save_path"], exist_ok=True)

    tensorboard_logger = TensorBoardLogger(config["save_path"], name="metrics")

    train_loader = get_dataloader(config)
    val_loader = get_dataloader(config, mode="val")

    model = TrainModel(config, train_loader, val_loader)
    summary(model.model, input_data=None)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config["save_path"],
        save_last=True,
        every_n_epochs=1,
        save_top_k=1,
        save_weights_only=True,
        save_on_train_epoch_end=False,
        **config["checkpoint"],
    )

    lr_callback = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[lr_callback, checkpoint_callback],
        logger=tensorboard_logger,
        log_every_n_steps=len(train_loader),
        **config["trainer"],
    )

    trainer.fit(model)


if __name__ == "__main__":
    config_path = ROOT_PATH / "configs/train.yaml"
    config = read_yaml(config_path)

    train(config)
