import os
import pickle
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

parser = argparse.ArgumentParser("YOLO")
parser.add_argument("-data", help="Data index", nargs='?', type=int, default=0)
parser.add_argument("-verbose", help="Verbose", nargs='?', type=bool, default=False)
args = parser.parse_args()

import tensorflow as tf
import data.cifar10 as data

from tqdm import *
from model.model import *
from attack.attack import *
from sklearn.metrics import roc_auc_score

if not os.path.isfile(f'{data.__FOLDER__}/cen/{args.data}.pickle'):
    print(f"Centralized: {args.data} --------------------------------------")
    strategy, AUTO = getStrategy()
    with strategy.scope():
        model, preprocess = model_factory()
        model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2),
                        loss = {'output': tf.keras.losses.SparseCategoricalCrossentropy()},
                        metrics = {"output": [tf.keras.metrics.SparseCategoricalAccuracy()]})

    trainData, inOutLabels = data.loadCenData(args.data, preprocess)
    model.fit(trainData, verbose = args.verbose, epochs = 100)

    miaData, validData = data.load(preprocess)
    validPred = model.predict(validData, verbose = args.verbose)
    print("Validation:", np.mean(validData.labels.flatten() == np.argmax(validPred, axis = 1)))

    MIAPred = model.predict(miaData, verbose = args.verbose)
    print("MIA:", np.mean(miaData.labels.flatten() == np.argmax(MIAPred, axis = 1)))

    with open(f'{data.__FOLDER__}/cen/{args.data}.pickle', 'wb') as handle:
        pickle.dump([MIAPred, inOutLabels], handle, protocol=pickle.HIGHEST_PROTOCOL)