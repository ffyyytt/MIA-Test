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
yPred = model.predict(miaData)
scores = miaEntropy(yPred)
print(roc_auc_score(miaLabels, scores))

shadowLabels = []
shadowPredicts = []
for i in trange(data.__N_SHADOW__):
    cenShadowTrain, _, shadowLabel = data.loadCenShadowTrain()
    with strategy.scope():
        model = model_factory()
        model.compile(optimizer = "sgd",
                    loss = {'output': tf.keras.losses.SparseCategoricalCrossentropy()},
                    metrics = {"output": [tf.keras.metrics.SparseCategoricalAccuracy()]})
        
    H = model.fit(cenShadowTrain, verbose = False, epochs = 100)
    shadowPredicts.append([model.predict(miaData), cenShadowTrain.labels])
    shadowLabels.append(shadowLabel)

scores = LiRAOnline(yPred, cenTrain.labels, shadowPredicts, shadowLabels)
print(roc_auc_score(miaLabels, scores))