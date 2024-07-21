import os
import cv2
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split

class TFDataGen(tf.keras.utils.Sequence):
    def __init__(self, images, labels, preprocess, batch_size, **kwargs):
        self.preprocess = preprocess
        self.labels = np.array(labels, dtype="float32")
        self.images = np.array(images, dtype="float32")
        self.ids = np.arange(len(self.labels))
        self.batch_size = batch_size
        super().__init__(**kwargs)

    def __len__(self):
        return len(self.images) // self.batch_size + int(len(self.images) % self.batch_size != 0)

    def __getitem__(self, index):
        ids = self.ids[index*self.batch_size: min((index+1)*self.batch_size, len(self.ids))]
        images = self.preprocess(self.images[ids])
        label = self.labels[ids]
        return {"image": images}, {"output": label}

__FOLDER__ = "AID"
__RANDOM__SEED__ = 1312
__N_CLASSES__ = 30
__N_SHADOW__ = 256
__N_CLIENTS__ = 4
__BATCH_SIZE__ = 32

def _loadAID():
    images = []
    labels = []
    labelset = {}
    imagePaths = glob.glob(os.path.expanduser("~")+"/data/AID/*/*")
    for file in imagePaths:
        try:
            if file.split("/")[-2] not in labelset:
                labelset[file.split("/")[-2]] = len(labelset)
            image = cv2.imread(file)   # reads an image in the BGR format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # BGR -> RGB
            images.append(image)
            labels.append([labelset[file.split("/")[-2]]])
        except:
            pass
    images = np.array(images)
    labels = np.array(labels)
    return train_test_split(images, labels, test_size=0.3, random_state=__RANDOM__SEED__)

X_train_aid, X_valid_aid, Y_train_aid, Y_valid_aid = _loadAID()

def load(preprocess):
    return TFDataGen(X_train_aid, Y_train_aid,  preprocess,  __BATCH_SIZE__), TFDataGen(X_valid_aid, Y_valid_aid,  preprocess,  __BATCH_SIZE__)

def loadCenData(idx, preprocess):
    images, labels = [], []
    inOutLabels = np.zeros([len(X_train_aid)])
    sss = StratifiedShuffleSplit(n_splits=__N_SHADOW__+1, test_size=0.5, random_state=__RANDOM__SEED__)
    for i, (_, indexes) in enumerate(sss.split(X_train_aid, np.argmax(Y_train_aid, axis=1))):
        if i == idx:
            images += X_train_aid[indexes].tolist()
            labels += Y_train_aid[indexes].tolist()
            inOutLabels[indexes] = 1
    return TFDataGen(images, labels,  preprocess,  __BATCH_SIZE__), inOutLabels

def loadFedData(idx, preprocess):
    images, labels = [], []
    inOutLabels = np.zeros([len(X_train_aid)])
    sss = StratifiedShuffleSplit(n_splits=__N_SHADOW__+1, test_size=0.5, random_state=__RANDOM__SEED__)
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