import os
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tqdm import *
from model.model import *
from attack.attack import *
import data.cifar10 as data

from sklearn.metrics import roc_auc_score

print("-------------------------------------LiRA-------------------------------------")
strategy, AUTO = getStrategy()
with strategy.scope():
    model, preprocess = model_factory()
    model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2),
                  loss = {'output': tf.keras.losses.SparseCategoricalCrossentropy()},
                  metrics = {"output": [tf.keras.metrics.SparseCategoricalAccuracy()]})
    
cenTrain, _ = data.loadCenTrain(preprocess)
miaData, miaLabels = data.loadMIAData(preprocess)
    
H = model.fit(cenTrain, verbose = False, epochs = 100)
yPred = model.predict(miaData, verbose = False)
scores = miaEntropy(yPred)
yPred = yPred[:, miaData.labels.flatten().astype(int)]
print(np.mean(np.argmax(yPred, axis=1) == miaData.labels), roc_auc_score(miaLabels, scores))

def doShadow():
    with strategy.scope():
        model, preprocess = model_factory()
        model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2),
                    loss = {'output': tf.keras.losses.SparseCategoricalCrossentropy()},
                    metrics = {"output": [tf.keras.metrics.SparseCategoricalAccuracy()]})
    cenShadowTrain, _, shadowLabel = data.loadCenShadowTrain(i, preprocess)
    H = model.fit(cenShadowTrain, verbose = False, epochs = 100)
    gc.collect()
    return model.predict(miaData, verbose = False)[:, miaData.labels.flatten().astype(int)], shadowLabel

shadowLabels = [0]*data.__N_SHADOW__
shadowPredicts = []
for i in trange(data.__N_SHADOW__):
    resutlt = doShadow()
    shadowPredicts.append(resutlt[0])
    shadowLabels[i] = resutlt[1]

scores = LiRAOnline(yPred, np.array(shadowPredicts), np.array(shadowLabels))
print(roc_auc_score(miaLabels, scores))