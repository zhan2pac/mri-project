#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from datasets import create_loader_dataset
from pl_models import TrainModel
import shutil
from loguru import logger


torch.cuda.empty_cache()


def load_config(config_path):
    with open(config_path, "r") as input_file:
        config = yaml.safe_load(input_file)

    return config


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
def train(args=None):

    config = load_config("/home/zhan/mrt_project/segmentation_template/configs/train.yaml")
    config["save_path"] = os.path.join(config["exp_path"], config["project"], config["exp_name"])

    check_dir(config["save_path"])
    os.makedirs(config["save_path"], exist_ok=True)

    tensorboard_logger = TensorBoardLogger(config["save_path"], name="metrics")

    train_loader, _, weights = create_loader_dataset(config)

    val_loader, _, _ = create_loader_dataset(config, mode="test")

    # if config['start_from'] is not None:
    #     model = TrainModel.load_from_checkpoint(
    #         os.path.join(config['start_from'], 'last.ckpt'),
    #         hparams_file=os.path.join(config['start_from'], 'metrics/version_0/hparams.yaml'),
    #         config = config,
    #         train_loader = train_loader,
    #         val_loader = val_loader,
    #         weights = weights
    #     )
    #     print(f"Model loaded from checkpoint: {config['start_from']}")
    # else:
    model = TrainModel(config, train_loader, val_loader, weights)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["save_path"],
        save_last=True,
        every_n_epochs=1,
        save_top_k=1,
        save_weights_only=True,
        save_on_train_epoch_end=False,
        **config["checkpoint"],
    )

    callbacks = [LearningRateMonitor(logging_interval="epoch"), checkpoint_callback]

    trainer = Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=callbacks,
        logger=tensorboard_logger,
        log_every_n_steps=len(train_loader),
        **config["trainer"],
    )
    trainer.fit(model)


if __name__ == "__main__":
    train()
