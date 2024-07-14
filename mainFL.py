import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

parser = argparse.ArgumentParser("YOLO")
parser.add_argument("-FL", help="Model backbone", nargs='?', type=str, default="FedAvg")
parser.add_argument("-method", help="0: Federated Learning, 1: Federated Feature", nargs='?', type=int, default=0)
args = parser.parse_args()

import numpy as np
import tensorflow as tf

from tqdm import *
from model import *
from attack import *
from myFL import *
import cifar10 as data

from sklearn.metrics import roc_auc_score

if args.method == 0:
    aggregate = avg_aggregate
else:
    aggregate = ft_aggregate


rounds = 10
localEpochs = 10
strategy, AUTO = getStrategy()
clientModels = []
with strategy.scope():
    serverModel, preprocess = model_factory()
    for i in range(data.__N_CLIENTS__):
        clientModels.append(model_factory()[0])
        
trainLoaders, validLoader = data.loadFLTrain(preprocess)
miaData, miaLabels = data.loadMIAData(preprocess)

doFL(clientModels, serverModel, trainLoaders, validLoader, localEpochs, aggregate, rounds, args.FL)
yPred = clientModels[0].predict(miaData)
scores = miaEntropy(yPred)
print("Accuracy:", np.mean(np.argmax(yPred, axis=1)==miaData.labels.flatten()), "AUC:", roc_auc_score(miaLabels, scores))