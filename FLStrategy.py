import flwr as fl
import tensorflow as tf
import keras.backend as K
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg
from flwr.server.strategy.fedavg import FedAvg

class MyFedAVG(FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        self.parameters_aggregated = parameters_aggregated
        return parameters_aggregated, metrics_aggregated

class MyFedProx(MyFedAVG):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        proximal_mu: float = 0.5,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.proximal_mu = proximal_mu

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedProx(accept_failures={self.accept_failures})"
        return rep

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.

        Sends the proximal factor mu to the clients
        """
        # Get the standard client/config pairs from the FedAvg super-class
        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        # Return client/config pairs with the proximal factor mu added
        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {**fit_ins.config, "proximal_mu": self.proximal_mu},
                ),
            )
            for client, fit_ins in client_config_pairs
        ]
    
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
                 mu=1e-4, clip_bounds=None, **kwargs):
        super(ProxSGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations      = K.variable(0, dtype='int64', name='iterations')
            self.epsilon_initial = K.variable(epsilon_initial, name='epsilon_initial')
            self.epsilon_decay   = K.variable(epsilon_decay, name='epsilon_decay')
            self.rho_initial     = K.variable(rho_initial, name='rho_initial')
            self.rho_decay       = K.variable(rho_decay, name='rho_decay')
            self.beta            = K.variable(beta, name='beta')
            self.mu              = mu
            self.clip_bounds     = clip_bounds

    def get_updates(self, loss, params):
        self.updates = [K.update_add(self.iterations, 1)]
        grads        = self.get_gradients(loss, params)
        iteration    = K.cast(self.iterations, K.dtype(self.epsilon_decay))
        epsilon      = self.epsilon_initial / ((iteration + 4) ** self.epsilon_decay) # the current lr for weights, see (8) of the paper
        rho          = self.rho_initial / ((iteration + 4) ** self.rho_decay) # the current lr for momentum, see (6) of the paper
        beta         = self.beta # the learning rate for the squared gradient, see Table I of the paper
        delta        = 1e-07 # see Table I of the paper
        
        if self.clip_bounds is not None:
            low = self.clip_bounds[0]
            up  = self.clip_bounds[1]

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        rs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + vs + rs

        for x, g, v, r in zip(params, grads, vs, rs):
            v_new = (1 - rho) * v + rho * g # update momentum according to (6) of the paper

            # define tau (same update rule as ADAM is adopted here, but other rules are also possible)
            r_new = beta * r + (1 - beta) * K.square(g) #see Table I of the paper
            tau   = K.sqrt(r_new / (1 - beta ** (iteration + 1))) + delta #see Table I of the paper

            '''solving the approximation subproblem (tailored to L1 norm and bound constraint)'''
            x_tmp = x - v_new / tau # the solution to the approximation subproblem without regularization/constraint
            
            if self.mu is not None: # apply soft-thresholding due to L1 norm
                mu_normalized = self.mu / tau
                x_hat = K.maximum(x_tmp - mu_normalized, 0) - K.maximum(-x_tmp - mu_normalized, 0)
            else: # if there is no L1 norm
                x_hat = x_tmp
                
            if self.clip_bounds is not None: # apply clipping due to the bound constraint
                x_hat = K.clip(x_hat, low, up)
            '''the approximation problem is solved'''
            
            x_new = x + epsilon * (x_hat - x) # update the weights according to (8) of the paper
            
            '''variable update'''
            self.updates.append(K.update(v, v_new))
            self.updates.append(K.update(r, r_new))
            self.updates.append(K.update(x, x_new))
        return self.updates
    
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
    
def trainFL(strategy, trainLoaders, validLoader, miaData, localEpochs, rounds, client_resources, data, model_factory):
    models = []
    with strategy.scope():
        initModel = model_factory()
        for i in range(data.__N_CLIENTS__):
            models.append(model_factory())
    def client_fn(cid):
        trainLoader = trainLoaders[int(cid)]
        model = models[int(cid)]
        return FlowerClient(cid, model, localEpochs, trainLoader, validLoader).to_client()
    
    FLStrategy = MyFedAVG(fraction_fit=1.,
                          fraction_evaluate=1.,
                          min_fit_clients=data.__N_CLIENTS__,
                          min_evaluate_clients=data.__N_CLIENTS__,
                          min_available_clients=data.__N_CLIENTS__,
                          initial_parameters=fl.common.ndarrays_to_parameters(initModel.trainable_variables))
    
    fl.simulation.start_simulation(client_fn=client_fn,
                                   num_clients=data.__N_CLIENTS__,
                                   config=fl.server.ServerConfig(num_rounds=rounds),
                                   strategy=FLStrategy,
                                   client_resources=client_resources)

    for i in range(len(initModel.trainable_variables)):
        initModel.trainable_variables[i].assign(FLStrategy.parameters_aggregated[i])
    yPred = initModel.predict(miaData, verbose = False)
    return yPred