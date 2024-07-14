import os
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tqdm import *
from model import *
from attack import *
import cifar10 as data

from sklearn.metrics import roc_auc_score

strategy, AUTO = getStrategy()
with strategy.scope():
    model, preprocess = model_factory()
    model.compile(optimizer = "sgd",
                  loss = {'output': tf.keras.losses.SparseCategoricalCrossentropy()},
                  metrics = {"output": [tf.keras.metrics.SparseCategoricalAccuracy()]})
    
cenTrain, _ = data.loadCenTrain(preprocess)
miaData, miaLabels = data.loadMIAData(preprocess)
    
H = model.fit(cenTrain, verbose = False, epochs = 100)
yPred = model.predict(miaData, verbose = False)
scores = miaEntropy(yPred)
yPred = yPred[:, miaData.labels.flatten().astype(int)]
print(np.mean(np.argmax(yPred, axis=1) == miaData.labels), roc_auc_score(miaLabels, scores))

shadowLabels = [0]*data.__N_SHADOW__
shadowPredicts = []
for i in trange(data.__N_SHADOW__):
    with strategy.scope():
        model, preprocess = model_factory()
        model.compile(optimizer = "sgd",
                      loss = {'output': tf.keras.losses.SparseCategoricalCrossentropy()},
                      metrics = {"output": [tf.keras.metrics.SparseCategoricalAccuracy()]})
    cenShadowTrain, _, shadowLabel = data.loadCenShadowTrain(i, preprocess)
    H = model.fit(cenShadowTrain, verbose = False, epochs = 100)
    shadowPredicts.append(model.predict(miaData, verbose = False)[:, miaData.labels.flatten().astype(int)])
    shadowLabels[i] = shadowLabel
    gc.collect()

scores = LiRAOnline(yPred, np.array(shadowPredicts), np.array(shadowLabels))
print(roc_auc_score(miaLabels, scores))