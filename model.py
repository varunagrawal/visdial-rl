import torch
import torch.nn as nn

from visdial.models.questioner import Questioner
from vqa.models.deeperlstm import DeeperLSTM

class PulpModel(nn.Module):
    def __init__(self, config, maps, image_feature_size):
        super(PulpModel, self).__init__()
        config["questioner"]["decoder"]["startToken"] = maps["word_to_wid"]["<START>"]
        config["questioner"]["decoder"]["endToken"] = maps["word_to_wid"]["<END>"]
        config["questioner"]["decoder"]["vocabSize"] = len(maps["word_to_wid"]) + 1
        config["questioner"]["encoder"]["vocabSize"] = len(maps["word_to_wid"]) + 1
        config["questioner"]["encoder"]["imgFeatureSize"] = image_feature_size
        config["answerer"]["vocab_size"] = len(maps["word_to_wid"]) + 1
        config["answerer"]["output_dim"] = len(maps["aid_to_ans"])
        self.questioner = Questioner(config["questioner"]["encoder"],
                config["questioner"]["decoder"],
                image_feature_size)
        self.answerer = DeeperLSTM(**config["answerer"])
        self.maps = maps
        self.aid_to_ans = maps["aid_to_ans"]

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
        answers_1_logits = self.answerer(self.image_1, questions, ques_lens)
        answers_2_logits = self.answerer(self.image_2, questions, ques_lens)
        answers_1 = self.aid_to_ans(torch.argmax(answers_1_logits, dim=2))
        answers_2 = self.aid_to_ans(torch.argmax(answers_2_logits, dim=2))

        dq_log_prob = self.discriminator(enc_state[0][-1], answers_1, answers_2)
        return q_log_probs, dq_log_prob

