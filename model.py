import torch
import torch.nn as nn

from visdial.models.questioner import Questioner
from vqa.models.deeperlstm import DeeperLSTM

class PulpModel(nn.Module):
    def __init__(self, config):
        self.questioner = Questioner(**config["questioner"])
        self.answerer = DeeperLSTM(**config["answerer"])

    def reset(self):
        pass

    def observe(self, round=-1, question=None, image_1=None, image_2=None):
        pass

    def forward(self):
        pass
