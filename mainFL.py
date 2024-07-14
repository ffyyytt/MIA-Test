import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf

from tqdm import *
from model import *
from attack import *
from myFL import *
import cifar10 as data

from sklearn.metrics import roc_auc_score

rounds = 25
localEpochs = 4
strategy, AUTO = getStrategy()
clientModels = []
with strategy.scope():
    serverModel, preprocess = model_factory()
    for i in range(data.__N_CLIENTS__):
        clientModels.append(model_factory()[0])
        clientModels[-1].compile(optimizer = "sgd", # ProxSGD(mu=config["proximal_mu"])
                                 loss = {'output': tf.keras.losses.SparseCategoricalCrossentropy()},
                                 metrics = {"output": [tf.keras.metrics.SparseCategoricalAccuracy()]})
trainLoaders, validLoader = data.loadFLTrain(preprocess)
miaData, miaLabels = data.loadMIAData(preprocess)

doFL(clientModels, serverModel, trainLoaders, validLoader, localEpochs, avg_aggregate, rounds)
yPred = serverModel.predict(miaData)
scores = miaEntropy(yPred)
print("Accuracy:", np.mean(np.argmax(yPred, axis=1)==miaData.labels.flatten()), "AUC:", roc_auc_score(miaLabels, scores))