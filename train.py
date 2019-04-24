import os
import gc
import random
import pprint
import yaml
import argparse
from pathlib import Path
from time import gmtime, strftime
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim

from vqa.models.deeperlstm import DeeperLSTM
from visdial.questioner import Questioner
from utils import utilities as utils
from utils.visualize import VisdomVisualize
# Placeholders
from dataloader import VarunLoader


parser = argparse.ArgumentParser(description="Train Discriminative Questioning bot.")
parser.add_argument('config', type=Path, help="YAML file for configuration.")
parser.add_argument("--data", type=Path, help="Dataset root directory.", default="./data")
parser.add_argument('--checkpoint', type=Path, help="Checkpoint to restart training from.")
parser.add_argument('--enable_visdom', action="store_true")

def create_data_loaders(config):
    data_loaders = dict()
    for split in config["dataset"]["splits"]:
        data_loaders[split] = VarunLoader(split)
    return data_loaders


def create_models(config, checkpoint=None):
    pass
    

def train(config, data_loaders, model, viz):
    pass

if __name__ == "__main__":
    args = parser.parse_args()
    with args["config"].open() as f:
        config = yaml.safe_load(f)

    if args["checkpoint"]:
        print("Loading checkpoint at {}".format(args["checkpoint"]))
        checkpoint = torch.load(args["checkpoint"])
    else:
        checkpoint = None

    if args["visdom_enabled"]:
        viz = VisdomVisualize()
    else:
        viz = VisdomVisualize(enable=False)

    data_loaders = create_data_loaders(config)
    models = create_models(config, checkpoint)

    train(config, data_loaders, models, viz)
