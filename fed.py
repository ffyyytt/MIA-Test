import os
import pickle
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

parser = argparse.ArgumentParser("YOLO")
parser.add_argument("-FL", help="Model backbone", nargs='?', type=str, default="FedAvg")
parser.add_argument("-method", help="0: Federated Learning, 1: Federated Feature", nargs='?', type=int, default=0)
parser.add_argument("-rounds", help="Rounds", nargs='?', type=int, default=5)
parser.add_argument("-epochs", help="Local epochs", nargs='?', type=int, default=20)
parser.add_argument("-data", help="Data index", nargs='?', type=int, default=0)
args = parser.parse_args()

import numpy as np
import tensorflow as tf

from tqdm import *
from model.model import *
from attack.attack import *
from model.FLStrategy import *
import data.cifar10 as data

from sklearn.metrics import roc_auc_score

if args.method == 0:
    aggregate = avg_aggregate
else:
    aggregate = ft_aggregate

if not os.path.isfile(f"{data.__FOLDER__}/{args.FL}{'FT'*args.method}/{args.data}.pickle"):
    print(f"{args.FL}{'FT'*args.method}: {args.data} --------------------------------------")
    strategy, AUTO = getStrategy()
    clientModels = []
    with strategy.scope():
        serverModel, preprocess = model_factory()
        for i in range(data.__N_CLIENTS__):
            clientModels.append(model_factory()[0])

    miaData, validData = data.load(preprocess)
    trainData, inOutLabels = data.loadFedData(args.data, preprocess)
    
    doFL(clientModels, serverModel, trainData, validData, args.epochs, aggregate, args.rounds, args.FL)

    validPred = clientModels[0].predict(validData, verbose = args.verbose)
    print("Validation:", np.mean(validData.labels.flatten() == np.argmax(validPred, axis = 1)))

    MIAPred = clientModels[0].predict(miaData, verbose = args.verbose)
    print("MIA:", np.mean(miaData.labels.flatten() == np.argmax(MIAPred, axis = 1)))

    with open(f"{data.__FOLDER__}/{args.FL}{'FT'*args.method}/{args.data}.pickle", 'wb') as handle:
        pickle.dump([MIAPred, inOutLabels], handle, protocol=pickle.HIGHEST_PROTOCOL)