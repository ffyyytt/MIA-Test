import glob
import pickle
from attack.attack import *
from keras.datasets import cifar10
from sklearn.metrics import roc_auc_score

predictions = []
inOutLabels = []
(X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()
Y_train = Y_train.flatten()
files = glob.glob("cifar10/cen/*.pickle")
for f in files:
    data = pickle.load(open(f,'rb'))
    predictions.append(data[0])
    inOutLabels.append(data[1])
    entropyScores = miaEntropy(data[0])
    entropyModScores = miaEntropyMod(data[0], Y_train)
    print(roc_auc_score(data[1], entropyScores))