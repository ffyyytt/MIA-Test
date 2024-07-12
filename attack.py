import numpy as np

def _log(probs, small_value=1e-30):
    return np.log(np.maximum(probs, small_value))

def miaEntropy(probs):
    return np.sum(np.multiply(probs, _log(probs)),axis=1)