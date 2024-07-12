import tensorflow as tf
from model import *
from attack import *
import cifar10 as data

from sklearn.metrics import roc_auc_score

auc = []
for i in range(10):
    strategy, AUTO = getStrategy()
    cenTrain = data.loadCenTrain()
    miaData, miaLabels = data.loadMIAData()

    with strategy.scope():
        model = model_factory()
        model.compile(optimizer = "sgd",
                    loss = {'output': tf.keras.losses.SparseCategoricalCrossentropy()},
                    metrics = {"output": [tf.keras.metrics.SparseCategoricalAccuracy()]})
        
    H = model.fit(cenTrain, verbose = False, epochs = 100)
    yPred = model.predict(miaData)
    scores = miaEntropy(yPred)
    auc.append(roc_auc_score(miaLabels, scores))
    print(auc)

print("Mean AUC:", np.mean(auc))