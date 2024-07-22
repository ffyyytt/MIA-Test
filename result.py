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
for folder in glob.glob("cifar10/*"):
    files = glob.glob(f"{folder}/*.pickle")
    print("n_shadow:", len(files))
    for f in tqdm(files):
        data = pickle.load(open(f,'rb'))
        predictions.append(data[1])
        inOutLabels.append(data[2])
        entropyScores = miaEntropy(data[1])
        entropyModScores = miaEntropyMod(data[1], Y_train_flatten)
        entropyAUC.append(roc_auc_score(data[2], entropyScores))
        entropyModAUC.append(roc_auc_score(data[2], entropyModScores))

    print(folder, "Entropy:", np.mean(entropyAUC))
    print(folder, "Entropy Mod:", np.mean(entropyModAUC))

# predictions = np.array(predictions)
# inOutLabels = np.array(inOutLabels)
# predictions = predictions[:, np.arange(Y_train.shape[0]), Y_train_flatten]
# inIdx = np.where(inOutLabels==1)
# outIdx = np.where(inOutLabels==0)
# inSet = predictions[inIdx]
# outSet = predictions[outIdx]
# print(inSet, np.mean(inSet))
# print(outSet, np.mean(outSet))
# print(predictions[0][inOutLabels[0]==1])
# print(predictions[0][inOutLabels[0]==0])
# inScores = probabilityNormalDistribution(inSet, predictions[0])
# outScores = probabilityNormalDistribution(outSet, predictions[0])
# LiRAScores = [inScore/outScore for inScore, outScore in zip(inScores, outScores)]
# print("Entropy Mod:", roc_auc_score(data[1], LiRAScores))