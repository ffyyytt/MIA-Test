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
Y_train = Y_train.flatten()
files = glob.glob("cifar10/cen/*.pickle")
print("n_shadow:", len(files))
for f in tqdm(files):
    data = pickle.load(open(f,'rb'))
    predictions.append(data[0])
    inOutLabels.append(data[1])
    entropyScores = miaEntropy(data[0])
    entropyModScores = miaEntropyMod(data[0], Y_train)
    entropyAUC.append(roc_auc_score(data[1], entropyScores))
    entropyModAUC.append(roc_auc_score(data[1], entropyModScores))

print("Entropy:", np.mean(entropyAUC))
print("Entropy Mod:", np.mean(entropyModAUC))

predictions = np.array(predictions)
inOutLabels = np.array(inOutLabels)

LiRAScores = LiRAOnline(np.max(predictions[0], axis=1), np.max(predictions[1:], axis=2), inOutLabels[1:], eps=1e-6)
print("Entropy Mod:", roc_auc_score(data[1], LiRAScores))