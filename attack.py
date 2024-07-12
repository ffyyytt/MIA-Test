import scipy
import numpy as np

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

def LiRAOnline(probs, labels, shadowPredicts, shadowLabels, eps=1e-6):
    _labels = labels.astype(int)
    _probs = probs[:, _labels]
    _shadowPredicts = np.array([shadowPredict[:, _labels] for shadowPredict in shadowPredicts])
    _shadowLabels = np.array(shadowLabels)
    return np.array([_LiRAOnline(_probs[i], _shadowPredicts[:, i], _shadowLabels[:, i], eps) for i in range(len(labels))])