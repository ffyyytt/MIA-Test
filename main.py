import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from tqdm import *
from model.model import *
from attack.attack import *
import data.cifar10 as data

from sklearn.metrics import roc_auc_score

print("-------------------------------------Centralized-------------------------------------")
strategy, AUTO = getStrategy()
auc = []
for i in trange(10):
    with strategy.scope():
        model, preprocess = model_factory()
    cenTrain, cenValid = data.loadCenTrain(preprocess)
    miaData, miaLabels = data.loadMIAData(preprocess)
        
    train(strategy, model, cenTrain)
    yPred = model.predict(miaData, verbose = False)
    scores = miaEntropy(yPred)
    auc.append(roc_auc_score(miaLabels, scores))
    print("Accuracy:", np.mean(np.argmax(yPred, axis=1)==miaData.labels.flatten()), "AUC:", roc_auc_score(miaLabels, scores))
    
print(auc)
print("Mean AUC:", np.mean(auc))