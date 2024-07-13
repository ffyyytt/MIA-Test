import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import flwr as fl
import numpy as np
import tensorflow as tf

from tqdm import *
from model import *
from attack import *
from FLStrategy import *
import cifar10 as data

from sklearn.metrics import roc_auc_score

rounds = 10
localEpochs = 10
strategy, AUTO = getStrategy()
with strategy.scope():
    initModel, preprocess = model_factory()
trainLoaders, validLoader = data.loadFLTrain(preprocess)
miaData, miaLabels = data.loadMIAData(preprocess)
client_resources = {"num_cpus": 8, "num_gpus": 1}

yPred = trainFL(strategy, initModel, trainLoaders, validLoader, miaData, localEpochs, rounds, client_resources, data.__N_CLIENTS__, model_factory)
scores = miaEntropy(yPred)
print("AUC:", roc_auc_score(miaLabels, scores))