import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


class TFDataGen(tf.keras.utils.Sequence):
    def __init__(self, images, labels, batch_size):
        self.labels = np.array(labels)
        self.images = np.array(images)
        self.ids = list(range(self.labels))
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return len(self.images) // self.batch_size + int(len(self.images) % self.batch_size != 0)

    def __getitem__(self, index):
        ids = self.ids[index*self.batch_size: min((index+1)*self.batch_size, len(self.ids))]
        images = self.images[ids]
        label = self.labels[ids]
        return {"image": images}, {"output": label}


__RANDOM__SEED__ = 1312
__CIFAR10_N_CLASSES__ = 10
__CIFAR10_N_SPLIT__ = 2
__CIFAR10_N_SHADOW__ = 256
__CIFAR10_TRAIN_SET__ = [0]
__CIFAR10_MEMBER_SET__ = [0]
__CIFAR10_NON_MEM_SET__ = [1]
__CIFAR10_BATCH_SIZE__ = 1024
def loadCenTrainCIFAR10():
    images, labels = [], []
    (X, Y), _ = cifar10.load_data()
    skf = StratifiedKFold(n_splits=__CIFAR10_N_SPLIT__, shuffle=True, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
        if i in __CIFAR10_TRAIN_SET__:
            images += X[test_index].tolist()
            labels += Y[test_index].tolist()
    return TFDataGen(images, labels,  __CIFAR10_BATCH_SIZE__)

def loadCenShadowTrainCIFAR10(idx):
    images, labels = [], []
    (X, Y), _ = cifar10.load_data()
    sss = StratifiedShuffleSplit(n_splits=__CIFAR10_N_SHADOW__, test_size=len(__CIFAR10_TRAIN_SET__)/__CIFAR10_N_SPLIT__, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(sss.split(X, np.argmax(Y, axis=1))):
        if i == idx:
            images += X[test_index].tolist()
            labels += Y[test_index].tolist()
    return TFDataGen(images, labels,  __CIFAR10_BATCH_SIZE__)

def loadMIADataCIFAR10():
    (X, Y), _ = cifar10.load_data()
    labels = np.zeros_like(Y)
    skf = StratifiedKFold(n_splits=__CIFAR10_N_SPLIT__, shuffle=True, random_state=__RANDOM__SEED__)
    for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
        if i in __CIFAR10_TRAIN_SET__:
            labels[test_index] = 1
    return TFDataGen(X, Y,  __CIFAR10_BATCH_SIZE__), labels