import os
import gc
import random
import pprint
from time import gmtime, strftime
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim

import options
from dataloader import VisDialDataset
from torch.utils.data import DataLoader
from eval_utils.rank_answerer import rankABot
from eval_utils.rank_questioner import rankQBot
from utils import utilities as utils
from utils.visualize import VisdomVisualize

#---------------------------------------------------------------------------------------
# Setup
#---------------------------------------------------------------------------------------

# Read the command line options
params = options.readCommandLine()

# Seed rng for reproducibility
random.seed(params['randomSeed'])
torch.manual_seed(params['randomSeed'])
if params['useGPU']:
    torch.cuda.manual_seed_all(params['randomSeed'])

# Setup dataloader
splits = ['train', 'val', 'test']

dataset = VisDialDataset(params, splits)

# Params to transfer from dataset
transfer = ['vocabSize', 'numOptions', 'numRounds']
for key in transfer:
    if hasattr(dataset, key):
        params[key] = getattr(dataset, key)

# Create save path and checkpoints folder
os.makedirs('checkpoints', exist_ok=True)
os.mkdir(params['savePath'])

# Loading Modules
parameters = []
aBot = None
qBot = None

# Loading Q-Bot
qBot, loadedParams, optim_state = utils.loadModel(params, 'qbot')
for key in loadedParams:
    params[key] = loadedParams[key]

# Filtering parameters which require a gradient update
parameters.extend(filter(lambda p: p.requires_grad, qBot.parameters()))

# Setup pytorch dataloader
# TODO Replace with VarunLoader
dataset.split = 'train'
dataloader = DataLoader(
    dataset,
    batch_size=params['batchSize'],
    shuffle=False,
    num_workers=params['numWorkers'],
    drop_last=True,
    collate_fn=dataset.collate_fn,
    pin_memory=False)

# Initializing visdom environment for plotting data
viz = VisdomVisualize(
    enable=bool(params['enableVisdom']),
    env_name=params['visdomEnv'],
    server=params['visdomServer'],
    port=params['visdomServerPort'])
pprint.pprint(params)
viz.addText(pprint.pformat(params, indent=4))

# Setup optimizer
if params['continue']:
    # Continuing from a loaded checkpoint restores the following
    startIterID = params['ckpt_iterid'] + 1  # Iteration ID
    lRate = params['ckpt_lRate']  # Learning rate
    print("Continuing training from iterId[%d]" % startIterID)
else:
    # Beginning training normally, without any checkpoint
    lRate = params['learningRate']
    startIterID = 0

optimizer = optim.Adam(parameters, lr=lRate)
if params['continue']:  # Restoring optimizer state
    print("Restoring optimizer state dict from checkpoint")
    optimizer.load_state_dict(optim_state)
runningLoss = None

mse_criterion = nn.MSELoss(reduction='none')

numIterPerEpoch = dataset.numDataPoints['train'] // params['batchSize']
print('\n%d iter per epoch.' % numIterPerEpoch)

if params['useCurriculum']:
    if params['continue']:
        rlRound = max(0, 9 - (startIterID // numIterPerEpoch))
    else:
        rlRound = params['numRounds'] - 1
else:
    rlRound = 0

#---------------------------------------------------------------------------------------
# Training
#---------------------------------------------------------------------------------------


def batch_iter(dataloader):
    for epochId in range(params['numEpochs']):
        for idx, batch in enumerate(dataloader):
            yield epochId, idx, batch


start_t = timer()
for epochId, idx, batch in batch_iter(dataloader):
    # Keeping track of iterId and epoch
    iterId = startIterID + idx + (epochId * numIterPerEpoch)
    epoch = iterId // numIterPerEpoch
    gc.collect()

    # Moving current batch to GPU, if available
    if dataset.useGPU:
        batch = {key: v.cuda() if hasattr(v, 'cuda') \
                                    else v for key, v in batch.items()}

    image = batch['img_feat']
    caption = batch['cap']
    captionLens = batch['cap_len']
    gtQuestions = batch['ques']
    gtQuesLens = batch['ques_len']
    gtAnswers = batch['ans']
    gtAnsLens = batch['ans_len']
    options = batch['opt']
    optionLens = batch['opt_len']
    gtAnsId = batch['ans_id']

    # Initializing optimizer and losses
    optimizer.zero_grad()
    loss = 0
    qBotLoss = 0
    aBotLoss = 0
    rlLoss = 0
    featLoss = 0
    qBotRLLoss = 0
    aBotRLLoss = 0
    predFeatures = None
    initialGuess = None
    numRounds = params['numRounds']
    # numRounds = 1 # Override for debugging lesser rounds of dialog

    # Setting training mode and observing captions, images where needed
    if qBot:
        qBot.train(), qBot.reset()
        qBot.observe(-1, caption=caption, captionLens=captionLens)

    initialGuess = qBot.predictImage()
    prevFeatDist = mse_criterion(initialGuess, image)
    featLoss += torch.mean(prevFeatDist)
    prevFeatDist = torch.mean(prevFeatDist,1)

    # Iterating over dialog rounds
    for round in range(numRounds):
        '''
        Loop over rounds of dialog. Currently three modes of training are
        supported:

            sl-qbot :
                Supervised pre-training of Q-Bot model using cross
                entropy loss with ground truth questions for the
                dialog model and mean squared error loss for image
                feature regression (i.e. image prediction)
        '''
        # Tracking components which require a forward pass
        # Q-Bot dialog model
        forwardQBot = (params['trainMode'] == 'sl-qbot'
                       or (params['trainMode'] == 'rl-full-QAf'
                           and round < rlRound))
        # Q-Bot feature regression network
        forwardFeatNet = (forwardQBot or params['trainMode'] == 'rl-full-QAf')

        # Questioner Forward Pass (dialog model)
        if forwardQBot:
            # Observe GT question for teacher forcing
            qBot.observe(
                round,
                ques=gtQuestions[:, round],
                quesLens=gtQuesLens[:, round])
            quesLogProbs = qBot.forward()
            # Cross Entropy (CE) Loss for Ground Truth Questions
            qBotLoss += utils.maskedNll(quesLogProbs,
                                        gtQuestions[:, round].contiguous())
            # Observe GT answer for updating dialog history
            qBot.observe(
                round,
                ans=gtAnswers[:, round],
                ansLens=gtAnsLens[:, round])

        # In order to stay true to the original implementation, the feature
        # regression network makes predictions before dialog begins and for
        # the first 9 rounds of dialog. This can be set to 10 if needed.
        MAX_FEAT_ROUNDS = 9

        # Questioner feature regression network forward pass
        if forwardFeatNet and round < MAX_FEAT_ROUNDS:
            # Make an image prediction after each round
            predFeatures = qBot.predictImage()
            featDist = mse_criterion(predFeatures, image)
            featDist = torch.mean(featDist)
            featLoss += featDist

            # Q-Bot makes a guess at the end of each round
            predFeatures = qBot.predictImage()

            # Computing reward based on Q-Bot's predicted image
            featDist = mse_criterion(predFeatures, image)
            featDist = torch.mean(featDist,1)

    # Loss coefficients
    featLoss = featLoss * params['featLossCoeff']
    # Averaging over rounds
    qBotLoss = (params['CELossCoeff'] * qBotLoss) / numRounds
    featLoss = featLoss / numRounds  #/ (numRounds+1)
    # Total loss
    loss = qBotLoss + featLoss
    loss.backward()
    optimizer.step()

    # Tracking a running average of loss
    if runningLoss is None:
        runningLoss = loss.data.item()
    else:
        runningLoss = 0.95 * runningLoss + 0.05 * loss.data.item()

    # Decay learning rate
    if lRate > params['minLRate']:
        for gId, group in enumerate(optimizer.param_groups):
            optimizer.param_groups[gId]['lr'] *= params['lrDecayRate']
        lRate *= params['lrDecayRate']
        if iterId % 10 == 0:  # Plot learning rate till saturation
            viz.linePlot(iterId, lRate, 'learning rate', 'learning rate')

    # Print every now and then
    if iterId % 10 == 0:
        end_t = timer()  # Keeping track of iteration(s) time
        curEpoch = float(iterId) / numIterPerEpoch
        timeStamp = strftime('%a %d %b %y %X', gmtime())
        printFormat = '[%s][Ep: %.2f][Iter: %d][Time: %5.2fs][Loss: %.3g]'
        printFormat += '[lr: %.3g]'
        printInfo = [
            timeStamp, curEpoch, iterId, end_t - start_t, loss.data.item(), lRate
        ]
        start_t = end_t
        print(printFormat % tuple(printInfo))

        # Update line plots
        if isinstance(qBotLoss, torch.Tensor):
            viz.linePlot(iterId, qBotLoss.data.item(), 'qBotLoss', 'train CE')
        if isinstance(featLoss, torch.Tensor):
            viz.linePlot(iterId, featLoss.data.item(), 'featLoss',
                     'train FeatureRegressionLoss')
        if isinstance(loss, torch.Tensor):
            viz.linePlot(iterId, loss.data.item(), 'loss', 'train loss')
        viz.linePlot(iterId, runningLoss, 'loss', 'running train loss')

    # Evaluate every epoch
    if iterId % (numIterPerEpoch // 1) == 0:
        # Keeping track of epochID
        curEpoch = float(iterId) / numIterPerEpoch
        epochId = (1.0 * iterId / numIterPerEpoch) + 1

        # Set eval mode
        if qBot:
            qBot.eval()

        if params['enableVisdom']:
            # Printing visdom environment name in terminal
            print("Currently on visdom env [%s]" % (params['visdomEnv']))

        # Mapping iteration count to epoch count
        viz.linePlot(iterId, epochId, 'iter x epoch', 'epochs')

        print('Performing validation...')

        if qBot:
            print("qBot Validation:")
            rankMetrics, roundMetrics = rankQBot(qBot, dataset, 'val')

            for metric, value in rankMetrics.items():
                viz.linePlot(
                    epochId, value, 'val - qBot', metric, xlabel='Epochs')

            viz.linePlot(iterId, epochId, 'iter x epoch', 'epochs')

            if 'logProbsMean' in rankMetrics:
                logProbsMean = params['CELossCoeff'] * rankMetrics[
                    'logProbsMean']
                viz.linePlot(iterId, logProbsMean, 'qBotLoss', 'val CE')

            if 'featLossMean' in rankMetrics:
                featLossMean = params['featLossCoeff'] * (
                    rankMetrics['featLossMean'])
                viz.linePlot(iterId, featLossMean, 'featLoss',
                             'val FeatureRegressionLoss')

            if 'logProbsMean' in rankMetrics and 'featLossMean' in rankMetrics:
                if params['trainMode'] == 'sl-qbot':
                    valLoss = logProbsMean + featLossMean
                    viz.linePlot(iterId, valLoss, 'loss', 'val loss')

    # Save the model after every epoch
    if iterId % numIterPerEpoch == 0:
        params['ckpt_iterid'] = iterId
        params['ckpt_lRate'] = lRate

        if qBot:
            saveFile = os.path.join(params['savePath'],
                                    'qbot_ep_%d.vd' % curEpoch)
            print('Saving model: ' + saveFile)
            utils.saveModel(qBot, optimizer, saveFile, params)
