import glob
import pickle
from attack.attack import *
from sklearn.metrics import roc_auc_score

predictions = []
inOutLabels = []
files = glob.glob("cifar10/cen/*.pickle")
for f in files:
    data = pickle.load(open(f,'rb'))
    predictions.append(data[0])
    inOutLabels.append(data[1])
    scores = miaEntropy(data[0])
    print(roc_auc_score(data[1], scores))