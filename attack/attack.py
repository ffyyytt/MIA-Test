import scipy
import numpy as np
from tqdm import *
from sklearn.metrics import roc_curve

def _log(probs, small_value=1e-30):
    return np.log(np.maximum(probs, small_value))

def miaEntropy(probs):
    return np.sum(np.multiply(probs, _log(probs)),axis=1)

def _log_value(probs, small_value=1e-30):
    return -np.log(np.maximum(probs, small_value))

def miaEntropyMod(probs, true_labels):
    log_probs = _log_value(probs)
    reverse_probs = 1-probs
    log_reverse_probs = _log_value(reverse_probs)
    modified_probs = np.copy(probs)
    modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
    modified_log_probs = np.copy(log_reverse_probs)
    modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
    return 1-np.sum(np.multiply(modified_probs, modified_log_probs),axis=1)

def probabilityNormalDistribution(data, p, eps=1e-100):
    if len(data) == 0:
        return 0.0
    mean = np.mean(data)
    std = max(np.std(data), eps)
    return scipy.stats.norm.cdf((p - mean) / std)

def _LiRAOnline(prob, shadowPredicts, shadowLabels, eps=1e-6):
    truthIdxs = np.where(shadowLabels==1)
    falseIdxs = np.where(shadowLabels==0)
    return probabilityNormalDistribution(shadowPredicts[truthIdxs].flatten(), prob)/max(probabilityNormalDistribution(shadowPredicts[falseIdxs].flatten(), prob), eps)

def LiRAOnline(probs, shadowPredicts, shadowLabels, eps=1e-6):
    results = [0]*len(probs)
    for i in trange(len(probs)):
        results[i] = _LiRAOnline(probs[i], shadowPredicts, shadowLabels, eps)
    return np.array(results)

def TPRatFPR(y_true, y_score, target_fpr = 0.001):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    tpr_at_target_fpr = tpr[np.where(fpr >= target_fpr)[0][0]]
    return tpr_at_target_fpr