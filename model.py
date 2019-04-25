import torch
import torch.nn as nn

from visdial.models.questioner import Questioner
from vqa.models.deeperlstm import DeeperLSTM

class PulpModel(nn.Module):
    def __init__(self, config):
        self.questioner = Questioner(**config["questioner"])
        self.answerer = DeeperLSTM(**config["answerer"])

        self.questions = dict()
        self.image_1 = None
        self.image_2 = None

    def reset(self):
        self.questions = dict()
        self.image_1 = None
        self.image_2 = None
        self.questioner.reset()

    def observe(self, round=-1, question=None, image_1=None, image_2=None):
        if isinstance(image_1, torch.Tensor):
            self.image_1 = image_1
            self.questioner.observe(round, image=image_1)
        
        if isinstance(image_2, torch.Tensor):
            self.image_2 = image_2

        if isinstance(question, torch.Tensor):
            self.questioner.observe(round, question)

    def forward(self):
        q_log_probs, enc_state = self.questioner.forward()
        questions, ques_lens = self.questioner.forwardDecode()
        answers_1 = self.answerer(self.image_1, questions, ques_lens)
        answers_2 = self.answerer(self.image_2, questions, ques_lens)

        dq_log_prob = self.discriminator(enc_state[0][-1], answers_1, answers_2)
        return q_log_probs, dq_log_prob

