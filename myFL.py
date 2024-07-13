import numpy as np

def doFL(client_models, server_model, trainLoaders, validLoader, local_epochs, aggregate_fn, rounds):
    for round in range(rounds):
        yPred = do_one_round(client_models, server_model, trainLoaders, validLoader, local_epochs, aggregate_fn)
    return yPred

def do_one_round(client_models, server_model, trainLoaders, validLoader, local_epochs, aggregate_fn):
    for i in range(len(server_model.trainable_variables)):
        for j in range(len(client_models)):
            client_models[j].trainable_variables[i].assign(server_model.trainable_variables[i])

    for i in range(len(client_models)):
        client_models[i].fit(trainLoaders[i], verbose=False, epochs=local_epochs)
    
    aggregate_fn(server_model, client_models)
    yPred = server_model.predict(validLoader, verbose=False)
    return yPred

def avg_aggregate(server_model, client_models):
    for i in range(len(server_model.trainable_variables)):
        server_model.trainable_variables[i].assign(sum([client_models[j].trainable_variables[i] for j in range(len(client_models))])/len(client_models))