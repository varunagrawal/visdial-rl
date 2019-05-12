import torch
import torch.nn as nn

from utils import utilities as utils


class Encoder(nn.Module):
    def __init__(
        self,
        vocabSize,
        embedSize,
        rnnHiddenSize,
        numLayers,
        useIm,
        imgEmbedSize,
        imgFeatureSize,
        numRounds,
        isAnswerer,
        dropout=0,
        startToken=None,
        endToken=None,
        **kwargs
    ):
        super(Encoder, self).__init__()
        self.vocabSize = vocabSize
        self.embedSize = embedSize
        self.rnnHiddenSize = rnnHiddenSize
        self.numLayers = numLayers
        assert self.numLayers > 1, "Less than 2 layers not supported!"
        if useIm:
            self.useIm = useIm if not useIm else "early"
        else:
            self.useIm = False
        self.imgEmbedSize = imgEmbedSize
        self.imgFeatureSize = imgFeatureSize
        self.numRounds = numRounds
        self.dropout = dropout
        self.isAnswerer = isAnswerer
        self.startToken = startToken
        self.endToken = endToken

        # modules
        self.wordEmbed = nn.Embedding(self.vocabSize, self.embedSize, padding_idx=0)

        # question encoder
        # image fuses early with words
        if self.useIm == "early":
            quesInputSize = self.embedSize + self.imgEmbedSize
            dialogInputSize = 3 * self.rnnHiddenSize
            self.imgNet = nn.Linear(self.imgFeatureSize, self.imgEmbedSize)
            self.imgEmbedDropout = nn.Dropout(0.5)
        elif self.useIm == "late":
            quesInputSize = self.embedSize
            dialogInputSize = 3 * self.rnnHiddenSize + self.imgEmbedSize
            self.imgNet = nn.Linear(self.imgFeatureSize, self.imgEmbedSize)
            self.imgEmbedDropout = nn.Dropout(0.5)
        else:
            dialogInputSize = self.rnnHiddenSize
        self.quesRNN = nn.LSTM(
            quesInputSize,
            self.rnnHiddenSize,
            self.numLayers,
            batch_first=True,
            dropout=0,
        )

        # history encoder
        self.factRNN = nn.LSTM(
            self.embedSize,
            self.rnnHiddenSize,
            self.numLayers,
            batch_first=True,
            dropout=0,
        )

        # im2fact
        self.im2hids = nn.Linear(self.imgEmbedSize, self.rnnHiddenSize)
        self.im2states = nn.Linear(self.imgEmbedSize, self.rnnHiddenSize)

        # dialog rnn
        self.dialogRNN = nn.LSTMCell(dialogInputSize, self.rnnHiddenSize)

    def reset(self):
        # batchSize is inferred from input
        self.batchSize = 0

        # Input data
        self.image = None
        self.imageEmbed = None

        self.questionTokens = []
        self.questionEmbeds = []
        self.questionLens = []

        self.answerTokens = []
        self.answerEmbeds = []
        self.answerLengths = []

        self.answerTokens2 = []
        self.answerEmbeds2 = []
        self.answerLengths2 = []

        # Hidden embeddings
        self.factEmbeds = []
        self.questionRNNStates = []
        self.dialogRNNInputs = []
        self.dialogHiddens = []

    def _initHidden(self):
        """Initial dialog rnn state - initialize with zeros"""
        # Dynamic batch size inference
        assert self.batchSize != 0, "Observe something to infer batch size."
        someTensor = self.dialogRNN.weight_hh.data
        h = someTensor.new(self.batchSize, self.dialogRNN.hidden_size).zero_()
        c = someTensor.new(self.batchSize, self.dialogRNN.hidden_size).zero_()
        return h, c

    def observe(
        self,
        round,
        image=None,
        ques=None,
        ans=None,
        ans2=None,
        quesLens=None,
        ansLens=None,
        ansLens2=None,
    ):
        """
        Store dialog input to internal model storage

        Note that all input sequences are assumed to be left-aligned (i.e.
        right-padded). Internally this alignment is changed to right-align
        for ease in computing final time step hidden states of each RNN
        """
        if image is not None:
            assert round == -1
            assert self.image is None
            self.image = image
            self.imageEmbed = None
            self.batchSize = len(self.image)
        if ques is not None:
            assert quesLens is not None, "Questions lengths required!"
            ques, quesLens = self.processSequence(ques, quesLens)
            if round == len(self.questionEmbeds):
                self.questionTokens.append(ques)
                self.questionLens.append(quesLens)
            elif round < len(self.questionEmbeds):
                self.questionTokens[round] = ques
                self.questionLens[round] = quesLens
                raise NotImplementedError
        if ans is not None:
            assert ansLens is not None, "Answer lengths required!"
            ans, ansLens = self.processSequence(ans, ansLens)
            if round == len(self.answerEmbeds):
                self.answerTokens.append(ans)
                self.answerLengths.append(ansLens)
            elif round < len(self.answerEmbeds):
                self.answerTokens[round] = ans
                self.answerLens[round] = ansLens
                raise NotImplementedError
        if ans2 is not None:
            assert ansLens2 is not None, "Answer lengths required!"
            ans2, ansLens2 = self.processSequence(ans2, ansLens2)
            if round == len(self.answerEmbeds2):
                self.answerTokens.append(ans2)
                self.answerLengths.append(ansLens2)
            elif round < len(self.answerEmbeds2):
                self.answerTokens2[round] = ans2
                self.answerLengths2[round] = ansLens2
                raise NotImplementedError

    def processSequence(self, seq, seqLen):
        """ Strip <START> and <END> token from a left-aligned sequence"""
        return seq[:, 1:], seqLen - 1

    def embedInputDialog(self):
        """
        Lazy embedding of input:
            Calling observe does not process (embed) any inputs. Since
            self.forward requires embedded inputs, this function lazily
            embeds them so that they are not re-computed upon multiple
            calls to forward in the same round of dialog.
        """
        # Embed image, occurs once per dialog
        if self.imageEmbed is None:
            self.imageEmbed = self.imgNet(self.imgEmbedDropout(self.image))
        # Embed questions
        while len(self.questionEmbeds) < len(self.questionTokens):
            idx = len(self.questionEmbeds)
            self.questionEmbeds.append(self.wordEmbed(self.questionTokens[idx]))
        # Embed answers
        while len(self.answerEmbeds) < len(self.answerTokens):
            idx = len(self.answerEmbeds)
            self.answerEmbeds.append(self.wordEmbed(self.answerTokens[idx]))
        while len(self.answerEmbeds2) < len(self.answerTokens2):
            idx = len(self.answerEmbeds2)
            self.answerEmbeds2.append(self.wordEmbed(self.answerTokens2[idx]))

    def embedFact(self, factIdx):
        """Embed facts i.e. image and round 0 or question-answer pair otherwise"""
        # Image
        if factIdx == 0:
            factEmbed, states = (
                self.im2hids(self.imageEmbed),
                [self.im2states(self.imageEmbed)],
            )
        # QAA triplets
        elif factIdx > 0:
            quesTokens, quesLens = (
                self.questionTokens[factIdx - 1],
                self.questionLens[factIdx - 1],
            )
            ansTokens, ansLens = (
                self.answerTokens[factIdx - 1],
                self.answerLengths[factIdx - 1],
            )
            ansTokens2, ansLens2 = (
                self.answerTokens2[factIdx - 1],
                self.answerLengths2[factIdx - 1],
            )

            qaTokens = utils.concatPaddedSequences(
                quesTokens, quesLens, ansTokens, ansLens, padding="right"
            )
            qaaTokens = utils.concatPaddedSequences(
                qaTokens, quesLens + ansLens, ansTokens2, ansLens2, padding="right"
            )
            qaa = self.wordEmbed(qaaTokens)
            qaaLens = quesLens + ansLens + ansLens2
            qaaEmbed, states = utils.dynamicRNN(
                self.factRNN, qaa, qaaLens, returnStates=True
            )
            factEmbed = qaaEmbed
        factRNNstates = states
        self.factEmbeds.append((factEmbed, factRNNstates))

    def embedQuestion(self, qIdx):
        """Embed questions"""
        quesIn = self.questionEmbeds[qIdx]
        quesLens = self.questionLens[qIdx]
        if self.useIm == "early":
            image = self.imageEmbed.unsqueeze(1).repeat(1, quesIn.size(1), 1)
            quesIn = torch.cat([quesIn, image], 2)
        qEmbed, states = utils.dynamicRNN(
            self.quesRNN, quesIn, quesLens, returnStates=True
        )
        quesRNNstates = states
        self.questionRNNStates.append((qEmbed, quesRNNstates))

    def concatDialogRNNInput(self, histIdx):
        currIns = [self.factEmbeds[histIdx][0]]
        if self.isAnswerer:
            currIns.append(self.questionRNNStates[histIdx][0])
        if self.useIm == "late":
            currIns.append(self.imageEmbed)
        hist_t = torch.cat(currIns, -1)
        self.dialogRNNInputs.append(hist_t)

    def embedDialog(self, dialogIdx):
        if dialogIdx == 0:
            hPrev = self._initHidden()
        else:
            hPrev = self.dialogHiddens[-1]
        inpt = self.dialogRNNInputs[dialogIdx]
        hNew = self.dialogRNN(inpt, hPrev)
        self.dialogHiddens.append(hNew)

    def forward(self):
        """
        Returns:
            A tuple of tensors (H, C) each of shape (batchSize, rnnHiddenSize)
            to be used as the initial Hidden and Cell states of the Decoder.
            See notes at the end on how (H, C) are computed.
        """
        round = len(self.questionTokens)

        # Lazily embed input Image, Captions, Questions and Answers
        self.embedInputDialog()

        # Lazy computation of internal hidden embeddings (hence the while loops)

        # Infer any missing facts
        while len(self.factEmbeds) <= round:
            factIdx = len(self.factEmbeds)
            self.embedFact(factIdx)

        # Embed any un-embedded questions
        while len(self.questionRNNStates) <= round:
            qIdx = len(self.questionRNNStates)
            self.embedQuestion(qIdx)

        # Concat facts and/or questions (i.e. history) for input to dialogRNN
        while len(self.dialogRNNInputs) <= round:
            histIdx = len(self.dialogRNNInputs)
            self.concatDialogRNNInput(histIdx)

        # Forward dialogRNN one step
        while len(self.dialogHiddens) <= round:
            dialogIdx = len(self.dialogHiddens)
            self.embedDialog(dialogIdx)

        # Latest dialogRNN hidden state
        dialogHidden = self.dialogHiddens[-1][0]

        """
        Return hidden (H_link) and cell (C_link) states as per the following rule:
        (Currently this is defined only for numLayers == 2)
        C_link == Fact encoding RNN cell state (factRNN)
        H_link ==
            Layer 0 : Fact encoding RNN hidden state (factRNN)
            Layer 1 : DialogRNN hidden state (dialogRNN)
        """
        factRNNstates = self.factEmbeds[-1][1]  # Latest factRNN states
        C_link = factRNNstates[1]
        H_link = factRNNstates[0][:-1]
        H_link = torch.cat([H_link, dialogHidden.unsqueeze(0)], 0)

        return H_link, C_link
