import torch
import torch.nn as nn
from visdial.models.agent import Agent
import visdial.models.encoders.hre as hre_enc
import visdial.models.decoders.gen as gen_dec
from utils import utilities as utils


class QAABot(Agent):
    def __init__(self, config, imgFeatureSize=0, verbose=1):
        """
            QAA-Bot Model

            Uses an encoder network for input sequences (questions, answers and
            history), a decoder network for generating a response (question),
            and answerer networks for answering its own questions.
        """
        super(QAABot, self).__init__()
        encoderParam = config["encoder"]
        decoderParam = config["decoder"]
        self.encType = encoderParam["type"]
        self.decType = decoderParam["type"]
        self.dropout = encoderParam["dropout"]
        self.rnnHiddenSize = encoderParam["rnn_hidden_size"]
        self.imgFeatureSize = imgFeatureSize
        encoderParam = encoderParam.copy()

        # Encoder
        if verbose:
            print("Encoder: " + self.encType)
            print("Decoder: " + self.decType)
        if "hre" in self.encType:
            self.encoder = hre_enc.create_hre(config["encoder"])
        else:
            raise Exception("Unknown encoder {}".format(self.encType))

        # Decoder
        if "gen" == self.decType:
            self.decoder = gen_dec.create_gen(config["decoder"])
        else:
            raise Exception("Unkown decoder {}".format(self.decType))

        # Answerers
        self.answerer = NotImplemented
        self.answerer2 = NotImplemented

        # Share word embedding parameters between encoder and decoder
        self.decoder.wordEmbed = self.encoder.wordEmbed

        # Create discriminator
        self.featureNet = nn.Linear(self.rnnHiddenSize, 1)
        self.featureNetInputDropout = nn.Dropout(0.5)

        # Initialize weights
        utils.initializeWeights(self.encoder)
        utils.initializeWeights(self.decoder)
        self.reset()

    def reset(self):
        """Delete dialog history."""
        self.questions = []
        self.encoder.reset()

    def freezeFeatNet(self):
        nets = [self.featureNet]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = False

    def observe(self, round, ques=None, image=None, image2=None, **kwargs):
        """
        Update Q-Bot percepts. See self.encoder.observe() in the corresponding
        encoder class definition (hre).
        """
        if ques is not None:
            if round == len(self.questions):
                self.questions.append(ques)
            elif round < len(self.questions):
                self.questions[round] = ques
                raise NotImplementedError
        if image is not None:
            self.image = image
        if image2 is not None:
            self.image2 = image2

        self.encoder.observe(round, ques=ques, image=image, **kwargs)

    def forward(self):
        """
        Forward pass the last observed question to compute its log
        likelihood under the current decoder RNN state.
        """
        encStates = self.encoder()
        if len(self.questions) == 0:
            raise Exception("Must provide question if not sampling one.")
        decIn = self.questions[-1]

        logProbs = self.decoder(encStates, inputSeq=decIn)
        return logProbs, encStates

    def forwardDecode(self, inference="sample", beamSize=1, maxSeqLen=20):
        """
        Decode a sequence (question) using either sampling or greedy inference.
        A question is decoded given current state (dialog history). This can
        be called at round 0 after the caption is observed, and at end of every
        round (after a response from A-Bot is observed).

        Arguments:
            inference : Inference method for decoding
                'sample' - Sample each word from its softmax distribution
                'greedy' - Always choose the word with highest probability
                           if beam size is 1, otherwise use beam search.
            beamSize  : Beam search width
            maxSeqLen : Maximum length of token sequence to generate
        """
        encStates = self.encoder()
        questions, quesLens = self.decoder.forwardDecode(
            encStates, maxSeqLen=maxSeqLen, inference=inference, beamSize=beamSize
        )
        ans, ansLens = self.answerer(self.image, questions, quesLens)
        ans2, ansLens2 = self.answerer2(self.image2, questions, quesLens)
        return questions, quesLens, ans, ansLens, ans2, ansLens2

    def discriminate(self):
        """
        Predict/guess an fc7 vector given the current conversation history. This can
        be called at round 0 after the caption is observed, and at end of every round
        (after a response from A-Bot is observed).
        """
        encState = self.encoder()
        # h, c from lstm
        h, c = encState
        return self.featureNet(self.featureNetInputDropout(h[-1]))

    def reinforce(self, reward):
        # Propogate reinforce function call to decoder
        return self.decoder.reinforce(reward)
