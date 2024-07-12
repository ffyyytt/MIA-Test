import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import flwr as fl
import numpy as np
import tensorflow as tf

from tqdm import *
from model import *
from attack import *
from FLStrategy import *
import cifar10 as data

from sklearn.metrics import roc_auc_score

rounds = 10
localEpochs = 10
strategy, AUTO = getStrategy()
trainLoaders, validLoader = data.loadFLTrain()
miaData, miaLabels = data.loadMIAData()
client_resources = {"num_cpus": 1, "num_gpus": 1}
models = []
with strategy.scope():
    initModel = model_factory()
    for i in range(data.__N_CLIENTS__):
        models.append(model_factory())

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, epochs, trainLoader, validLoader):
        self.cid = cid
        self.epochs = epochs
        self.trainLoader = trainLoader
        self.validLoader = validLoader

    def get_parameters(self, config):
        return self.net.trainable_variables
    
    def set_parameters(self, parameters):
        for i in range(len(self.net.trainable_variables)):
            self.net.trainable_variables[i].assign(parameters[i])

    def fit(self, parameters, config):
        if "proximal_mu" in config:
            optimizer = ProxSGD(mu=config["proximal_mu"])
        else:
            optimizer = "sgd"
        self.set_parameters(parameters)
        self.net.compile(optimizer = optimizer,
                         loss = {'output': tf.keras.losses.SparseCategoricalCrossentropy()},
                         metrics = {"output": [tf.keras.metrics.SparseCategoricalAccuracy()]})
        
        H = self.net.fit(self.trainLoader, verbose=False, epochs=self.epochs)
        return self.get_parameters(config), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        yPred = self.net.predict(self.validLoader)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(self.validLoader.labesl, yPred).mean()
        accuracy = np.mean(self.validLoader.labesl == np.argmax(yPred, axis=1))
        return float(loss), len(self.validLoader), {"accuracy": float(accuracy)}
        

def client_fn(cid):
    trainLoader = trainLoaders[int(cid)]
    model = models[int(cid)]
    return FlowerClient(cid, model, localEpochs, trainLoader, validLoader).to_client()

FLStrategy = MyFedAVG(
        fraction_fit=1.,
        fraction_evaluate=1.,
        min_fit_clients=data.__N_CLIENTS__,
        min_evaluate_clients=data.__N_CLIENTS__,
        min_available_clients=data.__N_CLIENTS__,
        initial_parameters=fl.common.ndarrays_to_parameters(initModel.trainable_variables),
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=data.__N_CLIENTS__,
    config=fl.server.ServerConfig(num_rounds=rounds),
    strategy=FLStrategy,
    client_resources=client_resources,
)

for i in range(len(initModel.trainable_variables)):
    initModel.trainable_variables[i].assign(FLStrategy.parameters_aggregated[i])
yPred = initModel.predict(miaData, verbose = False)
scores = miaEntropy(yPred)
print("AUC:", roc_auc_score(miaLabels, scores))