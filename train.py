import os
import gc
import random
import pprint
import yaml
import argparse
from pathlib import Path
from timeit import default_timer as timer
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim

from utils import utilities as utils
from utils.visualize import VisdomVisualize
from pulp_dataloader import get_dataloader
from model import PulpModel


parser = argparse.ArgumentParser(description="Train Discriminative Questioning bot.")
parser.add_argument('config', type=Path, help="YAML file for configuration.")
parser.add_argument("--data", type=Path, help="Dataset root directory.", default="./data")
parser.add_argument('--checkpoint', type=Path, help="Checkpoint to restart training from.")
parser.add_argument('--gpus', type=int, nargs='+', default=[-1], help="GPU IDs to use")

def create_data_loaders(config):
    data_loaders = dict()
    data_loaders["train"] = get_dataloader(
            Path(args.data, "coco_train_resnet152_pool5.pth"),
            Path(args.data, "v2_mscoco_train2014_annotations.json"),
            Path(args.data, "v2_OpenEnded_mscoco_train2014_questions.json"),
            Path(args.data, "v2_mscoco_train2014_complementary_pairs.json"),
            split="train",
            batch_size=config["train_batch_size"])
    maps= {"word_to_wid": data_loaders["train"].dataset.word_to_wid,
            "wid_to_word": data_loaders["train"].dataset.wid_to_word,
            "ans_to_aid": data_loaders["train"].dataset.ans_to_aid,
            "aid_to_ans": data_loaders["train"].dataset.aid_to_ans,
            "vocab": data_loaders["train"].dataset.vocab}
    data_loaders["val"] = get_dataloader(
            Path(args.data, "coco_train_resnet152_pool5.pth"),
            Path(args.data, "v2_mscoco_val2014_annotations.json"),
            Path(args.data, "v2_OpenEnded_mscoco_val2014_questions.json"),
            Path(args.data, "v2_mscoco_val2014_complementary_pairs.json"),
            split="val",
            maps=maps,
            batch_size=config["eval_batch_size"])
    return data_loaders, maps


def create_model(config, img_feat_size, maps, device, checkpoint=None):
    model = PulpModel(config, maps, img_feat_size)
    model.to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint)
    return model
    

def train_epoch(data_loader, model, criteria, optimizer, device):
    model.train()
    total_loss = 0.0
    samples = len(data_loader)

    for datum in data_loader:
        for key, value in datum.iteritems():
            value.to(device)
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


def eval_epoch(data_loader, model, criteria, device):
    model.eval()
    total_loss = 0.0
    samples = len(data_loader)

    with torch.no_grad():
        for datum in data_loader:
            for key, value in datum.iteritems():
                value.to(device)
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


def train(config, data_loaders, model, device):
    criteria = {"question": nn.CrossEntropyLoss(), "discriminative": nn.BCELoss()}
    if config["train"]["optimizer"] == "Adam":
        opt = optim.Adam
    if "optimizer_params" in config["train"]:
        optimizer = opt(model.parameters(), **config["train"]["optimizer_params"])
    else:
        optimizer = opt(model.parameters())
    # Start training
    for epoch in range(config["train"]["num_epochs"]):
        start_epoch = timer()
        train_loss = train_epoch(data_loaders["train"], criteria, optimizer, device)
        val_loss = eval_epoch(data_loaders["val"], model, criteria, device)
        print("[Epoch {:2d}] Training Loss: {:.4f}, Validation Loss: {:.4f}".format(train_loss, val_loss))
        save_path = Path(config["save_dir"], "epoch_{:3d}.pt".format(epoch))
        torch.save(model.state_dict(), save_path)
        stop_epoch = timer()
        print("Epoch elapsed time: {}, model saved at {}".format(timedelta(round(stop_epoch - start_epoch)), save_path))
        gc.collect()


if __name__ == "__main__":
    args = parser.parse_args()
    with args.config.open() as f:
        config = yaml.safe_load(f)

    if args.checkpoint:
        print("Loading checkpoint at {}".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
    else:
        checkpoint = None

    if len(args.gpus) > 1:
        device = "cuda"
        model = nn.DataParallel(model, device_ids=args.gpus)
    else:
        device = "cpu"

    data_loaders, maps = create_data_loaders(config["dataset"])
    img_feature_size = data_loaders["train"].dataset[0]["image_1"].size(1)
    config["answerer"]["image_dim"] = img_feature_size
    model = create_model(config, img_feature_size, maps, device, checkpoint)

    train(config, data_loaders, model, device)
    print("Training complete!")
