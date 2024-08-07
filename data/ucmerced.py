import os
import cv2
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split

class TFDataGen(tf.keras.utils.Sequence):
    def __init__(self, imagePaths, labels, preprocess, batch_size, **kwargs):
        self.preprocess = preprocess
        self.labels = np.array(labels, dtype="float32")
        self.imagePaths = np.array(imagePaths)
        self.ids = np.arange(len(self.labels))
        self.batch_size = batch_size
        super().__init__(**kwargs)

    def __len__(self):
        return len(self.imagePaths) // self.batch_size + int(len(self.imagePaths) % self.batch_size != 0)
    
    def readImages(self, ids):
        images = []
        for file in self.imagePaths[ids]:
            try:
                image = cv2.imread(file)   # reads an image in the BGR format
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # BGR -> RGB
                image = cv2.resize(image, (256, 256)) 
                images.append(image)
            except:
                pass
        return np.array(images, dtype="float32")

    def __getitem__(self, index):
        ids = self.ids[index*self.batch_size: min((index+1)*self.batch_size, len(self.ids))]
        images = self.preprocess(self.readImages(ids))
        label = self.labels[ids]
        return {"image": images}, {"output": label}

__FOLDER__ = "UCM"
__RANDOM__SEED__ = 1312
__N_CLASSES__ = 30
__N_SHADOW__ = 32
__N_CLIENTS__ = 10
__ROUNDS__ = 10
__LOCALEPOCHS__ = 2
__BATCH_SIZE__ = 16

def _loadAID():
    labels = []
    labelset = {}
    imagePaths = glob.glob(os.path.expanduser("~")+"/data/UCMerced_LandUse/*/*/*")
    for file in imagePaths:
        if file.split("/")[-2] not in labelset:
            labelset[file.split("/")[-2]] = len(labelset)
        labels.append([labelset[file.split("/")[-2]]])
    imagePaths = np.array(imagePaths)
    labels = np.array(labels)
    return train_test_split(imagePaths, labels, test_size=0.3, random_state=__RANDOM__SEED__)

X_train_aid, X_valid_aid, Y_train_aid, Y_valid_aid = _loadAID()

def load(preprocess):
    return TFDataGen(X_train_aid, Y_train_aid,  preprocess,  __BATCH_SIZE__), TFDataGen(X_valid_aid, Y_valid_aid,  preprocess,  __BATCH_SIZE__)

def loadCenData(idx, preprocess):
    images, labels = [], []
    inOutLabels = np.zeros([len(X_train_aid)])
    sss = StratifiedShuffleSplit(n_splits=__N_SHADOW__, test_size=0.5, random_state=__RANDOM__SEED__)
    for i, (_, indexes) in enumerate(sss.split(X_train_aid, np.argmax(Y_train_aid, axis=1))):
        if i == idx:
            images += X_train_aid[indexes].tolist()
            labels += Y_train_aid[indexes].tolist()
            inOutLabels[indexes] = 1
    return TFDataGen(images, labels,  preprocess,  __BATCH_SIZE__), inOutLabels

def loadFedData(idx, preprocess):
    images, labels = [], []
    inOutLabels = np.zeros([len(X_train_aid)])
    sss = StratifiedShuffleSplit(n_splits=__N_SHADOW__, test_size=0.5, random_state=__RANDOM__SEED__)
    for i, (_, indexes) in enumerate(sss.split(X_train_aid, np.argmax(Y_train_aid, axis=1))):
        if i == idx:
            images += X_train_aid[indexes].tolist()
            labels += Y_train_aid[indexes].tolist()
            inOutLabels[indexes] = 1

    images = np.array(images)
    labels = np.array(labels)
    data = []
    skf_fl = StratifiedKFold(n_splits=__N_CLIENTS__, shuffle=True, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(skf_fl.split(images, labels)):
        data.append(TFDataGen(images[test_index], labels[test_index],  preprocess,  __BATCH_SIZE__))
    return data, inOutLabels