import tensorflow as tf
from model import *
from attack import *
from cifar10 import *

from sklearn.metrics import roc_auc_score

strategy, AUTO = getStrategy()
cenTrainCIFAR10 = loadCenTrainCIFAR10()
miaData, miaLabels = loadMIADataCIFAR10()

with strategy.scope():
    model = model_factory()
    model.compile(optimizer = "sgd",
                  loss = {'output': tf.keras.losses.SparseCategoricalCrossentropy()},
                  metrics = {"output": [tf.keras.metrics.SparseCategoricalAccuracy()]})
    
H = model.fit(cenTrainCIFAR10, verbose = 1, epochs = 100)
yPred = model.predict(miaData)
scores = miaEntropy(yPred)
print(f"AUC: {roc_auc_score(miaLabels, scores)}")