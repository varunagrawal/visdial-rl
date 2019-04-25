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

from utils import utilities as utils
from utils.visualize import VisdomVisualize
from pulp_dataloader import get_dataloader


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
    

def train_epoch(data_loader, model, criteria, optimizer):
    model.train()
    total_loss = 0.0
    samples = len(data_loader)

    for datum in data_loader:
        loss = 0.0
        model.reset()
        model.zero_grad()
        model.observe(image_1=datum["image_1"])
        model.observe(image_2=datum["image_2"])
        for r in range(datum["questions"].size(1)):
            model.observe(round=r, question=datum["questions"][:, r])
            q_log_probs, dq_log_prob = model.forward()
            loss += criteria["question"](q_log_probs, datum["questions"][:, r])
            loss += criteria["discriminative"](dq_log_prob, datum["discriminant"] == r)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    average_loss = total_loss / samples
    return average_loss


def eval_epoch(data_loader, model, criteria):
    model.eval()
    total_loss = 0.0
    samples = len(data_loader)

    with torch.no_grad():
        for datum in data_loader:
            loss = 0.0
            model.reset()
            model.observe(image_1=datum["image_1"])
            model.observe(image_2=datum["image_2"])
            for r in range(datum["questions"].size(1)):
                model.observe(round=r, question=datum["questions"][:, r])
                q_log_probs, dq_log_prob = model.forward()
                loss += criteria["question"](q_log_probs, datum["questions"][:, r])
                loss += criteria["discriminative"](dq_log_prob, datum["discriminant"] == r)
            total_loss += loss.item()
    average_loss = total_loss / samples
    return average_loss


def train(config, data_loaders, model, viz):
    criteria = {"question": nn.CrossEntropyLoss(), "discriminative": nn.BCELoss()}
    if config["train"]["optimizer"] == "Adam":
        optimizer = optim.Adam(**config["train"]["optimizer_params"])
    # Start training
    for epoch in range(config["train"]["num_epochs"]):
        train_loss = train_epoch(data_loaders["train"], criteria, optimizer)
        print("Epoch {} training loss: {:.4f}".format(train_loss))
        val_loss = eval_epoch(data_loaders["val"], model)
        print("Validation loss: {:.4f}".format(val_loss))
        torch.save(model.state_dict(), Path(config["save_dir"], "epoch_{:3d}.pt".format(epoch)))
        gc.collect()


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
    print("Training complete!")
