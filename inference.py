#!/usr/bin/env python3
import argparse
import sys
import yaml
from pytorch_lightning import Trainer
from loguru import logger
from pytorch_lightning.strategies import DDPStrategy

from datasets.inference import create_inference_loader_dataset
from pl_models import TestModel
from utils import ROOT_PATH, read_yaml


@logger.catch
def inference(args=None):
    if args is None:
        args = sys.argv[1:]
        args = parse_args(args)
        config = read_yaml(args.config)
    else:
        config = args

    dataloader, dataset = create_inference_loader_dataset(
        config["annotations"],
        config["images"],
        config,
    )

    model = TestModel(config, dataloader, dataset)

    tester = Trainer(strategy=DDPStrategy(find_unused_parameters=False), logger=False, **config["trainer"])
    tester.test(model)


if __name__ == "__main__":
    config_path = ROOT_PATH / "configs/inference.yaml"
    config = read_yaml(config_path)

    inference(config)
