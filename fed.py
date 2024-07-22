import os
import pickle
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

parser = argparse.ArgumentParser("YOLO")
parser.add_argument("-FL", help="Model backbone", nargs='?', type=str, default="FedAvg")
parser.add_argument("-method", help="0: Federated Learning, 1: Federated Feature", nargs='?', type=int, default=0)
parser.add_argument("-rounds", help="Rounds", nargs='?', type=int, default=10)
parser.add_argument("-epochs", help="Local epochs", nargs='?', type=int, default=10)
parser.add_argument("-dataset", help="Dataset", nargs='?', type=str, default="cifar10")
parser.add_argument("-index", help="Data index", nargs='?', type=int, default=0)
parser.add_argument("-verbose", help="Verbose", nargs='?', type=bool, default=False)
args = parser.parse_args()

import numpy as np
import tensorflow as tf

from tqdm import *
from model.model import *
from attack.attack import *
from model.FLStrategy import *

if args.dataset == "cifar10":
    import data.cifar10 as data
elif args.dataset == "AID":
    import data.aid as data
else:
    print("Dataset not found")
    import data.cifar10 as data

if args.method == 1:
    aggregate = ft_aggregate
else:
    aggregate = avg_aggregate

if not os.path.isfile(f"{data.__FOLDER__}/{args.FL}{'FT'*args.method}/{args.index}.pickle"):
    print(f"{args.FL}{'FT'*args.method}: {args.index} --------------------------------------")
    strategy, AUTO = getStrategy()
    clientModels = []
    with strategy.scope():
        serverModel, preprocess = model_factory(n_classes=data.__N_CLASSES__)
        for i in range(data.__N_CLIENTS__):
            clientModels.append(model_factory(n_classes=data.__N_CLASSES__)[0])

    miaData, validData = data.load(preprocess)
    trainData, inOutLabels = data.loadFedData(args.index, preprocess)

    if args.method == 1:
        for i in range(len(clientModels)):
            doFT(clientModels[i], trainData[i])
    
    doFL(strategy, clientModels, serverModel, trainData, validData, args.epochs, aggregate, args.rounds, args.FL, args)
    
    if args.method != 2:
        validPred = clientModels[0].predict(validData, verbose = args.verbose)
        print("Validation:", np.mean(validData.labels.flatten() == np.argmax(validPred, axis = 1)))

        MIAPred = clientModels[0].predict(miaData, verbose = args.verbose)
        print("MIA:", np.mean(miaData.labels.flatten() == np.argmax(MIAPred, axis = 1)))
    else:
        modelFT, knn = buildkNN(clientModels[0], trainData)
        validPred = knn.predict_proba(modelFT.predict(validData, verbose = args.verbose))
        print("Validation:", np.mean(validData.labels.flatten() == np.argmax(validPred, axis = 1)))

        MIAPred = knn.predict_proba(modelFT.predict(miaData, verbose = args.verbose))
        print("MIA:", np.mean(miaData.labels.flatten() == np.argmax(MIAPred, axis = 1)))

    with open(f"{data.__FOLDER__}/{args.FL}{'FT'*args.method}/{args.index}.pickle", 'wb') as handle:
        pickle.dump([MIAPred, inOutLabels], handle, protocol=pickle.HIGHEST_PROTOCOL)