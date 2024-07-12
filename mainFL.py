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
trainLoaders, validLoader = data.loadFLTrain()
miaData, miaLabels = data.loadMIAData()
client_resources = {"num_cpus": 1, "num_gpus": 1}

yPred = trainFL(strategy, trainLoaders, validLoader, miaData, localEpochs, rounds, client_resources, data, model_factory)
scores = miaEntropy(yPred)
print("AUC:", roc_auc_score(miaLabels, scores))