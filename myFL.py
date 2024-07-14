import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import *

def doFL(client_models, server_model, trainLoaders, validLoader, local_epochs, aggregate_fn, rounds):
    for round in trange(rounds):
        yPred = do_one_round(client_models, server_model, trainLoaders, validLoader, local_epochs, aggregate_fn)
        print(np.mean(np.argmax(yPred, axis=1) == validLoader.labels.flatten()))
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

def ft_aggregate(server_model, client_models):
    server_model.trainable_variables[-1].assign(sum([client_models[j].trainable_variables[-1] for j in range(len(client_models))])/len(client_models))

class ProxSGD(tf.keras.optimizers.Optimizer):
    """ProxSGD optimizer (tailored for L1-norm regularization and bound constraint), proposed in
    ProxSGD: Training Structured Neural Networks under Regularization and Constraints, ICLR 2020
    URL: https://openreview.net/forum?id=HygpthEtvr
    # Arguments
        epsilon_initial: float >= 0. initial learning rate for weights.
        epsilon_decay  : float >= 0. learning rate (for weights) decay over each update.
        rho_initial    : float >= 0. initial learning rate for momentum.
        rho_decay      : float >= 0. learning rate (for momentum) decay over each update.
        beta           : float >= 0. second momentum parameter.
        mu             : float >= 0. regularization parameter for L1 norm.
        clipping_bound : float.      A vector including lower bound and upper bound.
    """

    def __init__(self, epsilon_initial=0.06, epsilon_decay=0.5, rho_initial=0.9, rho_decay=0.5, beta=0.999,
                 mu=1e-4, clip_bounds=None, name='ProxSGD', **kwargs):
        super(ProxSGD, self).__init__(name, **kwargs)
        self.epsilon_initial = epsilon_initial
        self.epsilon_decay = epsilon_decay
        self.rho_initial = rho_initial
        self.rho_decay = rho_decay
        self.beta = beta
        self.mu = mu
        self.clip_bounds = clip_bounds
        self.iterations = tf.Variable(0, dtype='int64', name='iterations')
        
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'v')
            self.add_slot(var, 'r')
        
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = apply_state[(var_device, var_dtype)]
        
        epsilon = self.epsilon_initial / ((self.iterations + 4) ** self.epsilon_decay)
        rho = self.rho_initial / ((self.iterations + 4) ** self.rho_decay)
        beta = self.beta
        delta = 1e-07

        v = self.get_slot(var, 'v')
        r = self.get_slot(var, 'r')

        v_new = (1 - rho) * v + rho * grad
        r_new = beta * r + (1 - beta) * tf.square(grad)
        tau = tf.sqrt(r_new / (1 - tf.pow(beta, tf.cast(self.iterations + 1, tf.float32)))) + delta

        x_tmp = var - v_new / tau

        if self.mu is not None:
            mu_normalized = self.mu / tau
            x_hat = tf.maximum(x_tmp - mu_normalized, 0) - tf.maximum(-x_tmp - mu_normalized, 0)
        else:
            x_hat = x_tmp

        if self.clip_bounds is not None:
            low = self.clip_bounds[0]
            up = self.clip_bounds[1]
            x_hat = tf.clip_by_value(x_hat, low, up)

        var_update = var + epsilon * (x_hat - var)

        self.iterations.assign_add(1)
        v.assign(v_new)
        r.assign(r_new)
        var.assign(var_update)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # Not implemented for sparse updates
        raise NotImplementedError("Sparse gradient updates are not supported.")
        
    def get_config(self):
        config = super(ProxSGD, self).get_config()
        config.update({
            'epsilon_initial': self.epsilon_initial,
            'epsilon_decay': self.epsilon_decay,
            'rho_initial': self.rho_initial,
            'rho_decay': self.rho_decay,
            'beta': self.beta,
            'mu': self.mu,
            'clip_bounds': self.clip_bounds
        })
        return config

    def update_step(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = apply_state[(var_device, var_dtype)]

        epsilon = self.epsilon_initial / ((self.iterations + 4) ** self.epsilon_decay)
        rho = self.rho_initial / ((self.iterations + 4) ** self.rho_decay)
        beta = self.beta
        delta = 1e-07

        v = self.get_slot(var, 'v')
        r = self.get_slot(var, 'r')

        v_new = (1 - rho) * v + rho * grad
        r_new = beta * r + (1 - beta) * tf.square(grad)
        tau = tf.sqrt(r_new / (1 - tf.pow(beta, tf.cast(self.iterations + 1, tf.float32)))) + delta

        x_tmp = var - v_new / tau

        if self.mu is not None:
            mu_normalized = self.mu / tau
            x_hat = tf.maximum(x_tmp - mu_normalized, 0) - tf.maximum(-x_tmp - mu_normalized, 0)
        else:
            x_hat = x_tmp

        if self.clip_bounds is not None:
            low = self.clip_bounds[0]
            up = self.clip_bounds[1]
            x_hat = tf.clip_by_value(x_hat, low, up)

        var_update = var + epsilon * (x_hat - var)

        self.iterations.assign_add(1)
        v.assign(v_new)
        r.assign(r_new)
        var.assign(var_update)