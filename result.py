import glob
import pickle
from tqdm import *
from attack.attack import *
from keras.datasets import cifar10
from sklearn.metrics import roc_auc_score

entropyAUC = []
entropyModAUC = []
predictions = []
inOutLabels = []
(X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()
Y_train_flatten = Y_train.flatten()
files = glob.glob("cifar10/cen/*.pickle")
print("n_shadow:", len(files))
for f in tqdm(files):
    data = pickle.load(open(f,'rb'))
    predictions.append(data[0])
    inOutLabels.append(data[1])
    entropyScores = miaEntropy(data[0])
    entropyModScores = miaEntropyMod(data[0], Y_train_flatten)
    entropyAUC.append(roc_auc_score(data[1], entropyScores))
    entropyModAUC.append(roc_auc_score(data[1], entropyModScores))

print("Entropy:", np.mean(entropyAUC))
print("Entropy Mod:", np.mean(entropyModAUC))

predictions = np.array(predictions)
inOutLabels = np.array(inOutLabels)

inIdx = np.where(inOutLabels==1)
outIdx = np.where(inOutLabels==0)
inSet = predictions[inIdx].flatten()
outSet = predictions[outIdx].flatten()
print(inSet.shape)
print(outSet.shape)
inScores = probabilityNormalDistribution(inSet, predictions[0][np.arange(Y_train.shape[0]), Y_train_flatten])
outScores = probabilityNormalDistribution(outSet, predictions[0][np.arange(Y_train.shape[0]), Y_train_flatten])
LiRAScores = [inScore/outScore for inScore, outScore in zip(inScores, outScores)]
print("Entropy Mod:", roc_auc_score(data[1], LiRAScores))