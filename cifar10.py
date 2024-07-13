import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


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


__RANDOM__SEED__ = 1312
__N_CLASSES__ = 10
__N_SPLIT__ = 2
__N_SHADOW__ = 8
__N_CLIENTS__ = 10
__TRAIN_SET__ = [0]
__MEMBER_SET__ = [0]
__NON_MEM_SET__ = [1]
__BATCH_SIZE__ = 128
def loadCenTrain(preprocess):
    images, labels = [], []
    (X, Y), (X_valid, Y_valid) = cifar10.load_data()
    skf = StratifiedKFold(n_splits=__N_SPLIT__, shuffle=True, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
        if i in __TRAIN_SET__:
            images += X[test_index].tolist()
            labels += Y[test_index].tolist()
    return TFDataGen(images, labels,  preprocess,  __BATCH_SIZE__), TFDataGen(X_valid, Y_valid,  preprocess,  __BATCH_SIZE__)

def loadFLTrain(preprocess):
    images, labels = [], []
    (X, Y), (X_valid, Y_valid) = cifar10.load_data()
    skf = StratifiedKFold(n_splits=__N_SPLIT__, shuffle=True, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
        if i in __TRAIN_SET__:
            images += X[test_index].tolist()
            labels += Y[test_index].tolist()

    images = np.array(images)
    labels = np.array(labels)
    data = []
    skf_fl = StratifiedKFold(n_splits=__N_CLIENTS__, shuffle=True, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(skf_fl.split(images, labels)):
        data.append(TFDataGen(images[test_index], labels[test_index],  preprocess,  __BATCH_SIZE__))
    return data, TFDataGen(X_valid, Y_valid,  preprocess,  __BATCH_SIZE__)

def loadCenShadowTrain(idx, preprocess):
    images, labels = [], []
    (X, Y), (X_valid, Y_valid) = cifar10.load_data()
    shadowLabel = np.zeros([len(X)])
    sss = StratifiedShuffleSplit(n_splits=__N_SHADOW__, test_size=len(__TRAIN_SET__)/__N_SPLIT__, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(sss.split(X, np.argmax(Y, axis=1))):
        if i == idx:
            images += X[test_index].tolist()
            labels += Y[test_index].tolist()
            shadowLabel[test_index] = 1
    return TFDataGen(images, labels,  preprocess,  __BATCH_SIZE__), TFDataGen(X_valid, Y_valid,  preprocess,  __BATCH_SIZE__), shadowLabel

def loadFLShadowTrain(idx, preprocess):
    images, labels = [], []
    (X, Y), (X_valid, Y_valid) = cifar10.load_data()
    shadowLabel = np.zeros([len(X)])
    sss = StratifiedShuffleSplit(n_splits=__N_SHADOW__, test_size=len(__TRAIN_SET__)/__N_SPLIT__, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(sss.split(X, np.argmax(Y, axis=1))):
        if i == idx:
            images += X[test_index].tolist()
            labels += Y[test_index].tolist()
            shadowLabel[test_index] = 1

    images = np.array(images)
    labels = np.array(labels)
    data = []
    skf_fl = StratifiedKFold(n_splits=__N_CLIENTS__, shuffle=True, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(skf_fl.split(images, labels)):
        data.append(TFDataGen(images[test_index], labels[test_index],  preprocess,  __BATCH_SIZE__))
    return data, TFDataGen(X_valid, Y_valid,  preprocess,  __BATCH_SIZE__), shadowLabel

def loadMIAData(preprocess):
    (X, Y), _ = cifar10.load_data()
    labels = np.zeros_like(Y)
    skf = StratifiedKFold(n_splits=__N_SPLIT__, shuffle=True, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
        if i in __TRAIN_SET__:
            labels[test_index] = 1
    return TFDataGen(X, Y, preprocess,  __BATCH_SIZE__), labels