import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from tqdm import *
from model import *
from attack import *
import cifar10 as data

from sklearn.metrics import roc_auc_score

strategy, AUTO = getStrategy()
auc = []
for i in trange(10):
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
    auc.append(roc_auc_score(miaLabels, scores))
    
print(auc)
print("Mean AUC:", np.mean(auc))