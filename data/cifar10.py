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

__FOLDER__ = "cifar10"
__RANDOM__SEED__ = 1312
__N_CLASSES__ = 10
__N_SHADOW__ = 32
__N_CLIENTS__ = 4
__BATCH_SIZE__ = 32
(X_train_cifar10, Y_train_cifar10), (X_valid_cifar10, Y_valid_cifar10) = cifar10.load_data()

def load(preprocess):
    return TFDataGen(X_train_cifar10, Y_train_cifar10,  preprocess,  __BATCH_SIZE__), TFDataGen(X_valid_cifar10, Y_valid_cifar10,  preprocess,  __BATCH_SIZE__)

def loadCenData(idx, preprocess):
    images, labels = [], []
    inOutLabels = np.zeros([len(X_train_cifar10)])
    sss = StratifiedShuffleSplit(n_splits=__N_SHADOW__, test_size=0.5, random_state=__RANDOM__SEED__)
    for i, (_, indexes) in enumerate(sss.split(X_train_cifar10, np.argmax(Y_train_cifar10, axis=1))):
        if i == idx:
            images += X_train_cifar10[indexes].tolist()
            labels += Y_train_cifar10[indexes].tolist()
            inOutLabels[indexes] = 1
    return TFDataGen(images, labels,  preprocess,  __BATCH_SIZE__), inOutLabels

def loadFedData(idx, preprocess):
    images, labels = [], []
    inOutLabels = np.zeros([len(X_train_cifar10)])
    sss = StratifiedShuffleSplit(n_splits=__N_SHADOW__, test_size=0.5, random_state=__RANDOM__SEED__)
    for i, (_, indexes) in enumerate(sss.split(X_train_cifar10, np.argmax(Y_train_cifar10, axis=1))):
        if i == idx:
            images += X_train_cifar10[indexes].tolist()
            labels += Y_train_cifar10[indexes].tolist()
            inOutLabels[indexes] = 1

    images = np.array(images)
    labels = np.array(labels)
    data = []
    skf_fl = StratifiedKFold(n_splits=__N_CLIENTS__, shuffle=True, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(skf_fl.split(images, labels)):
        data.append(TFDataGen(images[test_index], labels[test_index],  preprocess,  __BATCH_SIZE__))
    return data, inOutLabels