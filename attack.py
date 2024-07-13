import scipy
import numpy as np
from tqdm import *

def _log(probs, small_value=1e-30):
    return np.log(np.maximum(probs, small_value))

def miaEntropy(probs):
    return np.sum(np.multiply(probs, _log(probs)),axis=1)

def probabilityNormalDistribution(data, p, eps=1e-6):
    if len(data) == 0:
        return 0.0
    mean = np.mean(data)
    std = max(np.std(data), eps)
    return scipy.stats.norm.cdf((p - mean) / std)

def _LiRAOnline(prob, shadowPredicts, shadowLabels, eps=1e-6):
    truthIdxs = np.where(shadowLabels[:len(shadowPredicts)]==1)[0]
    falseIdxs = np.where(shadowLabels[:len(shadowPredicts)]==0)[0]
    return probabilityNormalDistribution(shadowPredicts[truthIdxs], prob)/max(probabilityNormalDistribution(shadowPredicts[falseIdxs], prob), eps)

def LiRAOnline(probs, shadowPredicts, shadowLabels, eps=1e-6):
    results = [0]*len(probs)
    for i in trange(len(probs)):
        results[i] = _LiRAOnline(probs[i], shadowPredicts[:, i], shadowLabels[:, i], eps)
    return np.array(results)