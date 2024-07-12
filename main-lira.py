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
cenTrain, _ = data.loadCenTrain()
miaData, miaLabels = data.loadMIAData()

with strategy.scope():
    model = model_factory()
    model.compile(optimizer = "sgd",
                  loss = {'output': tf.keras.losses.SparseCategoricalCrossentropy()},
                  metrics = {"output": [tf.keras.metrics.SparseCategoricalAccuracy()]})
    
H = model.fit(cenTrain, verbose = False, epochs = 100)
yPred = model.predict(miaData, verbose = False)
scores = miaEntropy(yPred)
yPred = yPred[:, cenTrain.labels.astype(int)]
print(np.mean(np.argmax(yPred, axis=1) == cenTrain.labels), roc_auc_score(miaLabels, scores))

shadowLabels = [0]*data.__N_SHADOW__
shadowPredicts = []
for i in trange(data.__N_SHADOW__):
    cenShadowTrain, _, shadowLabel = data.loadCenShadowTrain(i)
    with strategy.scope():
        model = model_factory()
        model.compile(optimizer = "sgd",
                      loss = {'output': tf.keras.losses.SparseCategoricalCrossentropy()},
                      metrics = {"output": [tf.keras.metrics.SparseCategoricalAccuracy()]})
        
    H = model.fit(cenShadowTrain, verbose = False, epochs = 100)
    shadowPredicts.append(model.predict(miaData, verbose = False)[:, cenTrain.labels.astype(int)])
    shadowLabels[i] = shadowLabel
    gc.collect()

scores = LiRAOnline(yPred, np.array(shadowPredicts), np.array(shadowLabels))
print(roc_auc_score(miaLabels, scores))