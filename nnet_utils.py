import numpy as np

def softmax(x,axis=1):
    #Numerically stable softmax
    return np.exp(log_softmax(x,axis))

def log_softmax(x,axis=1):
    return x - np.expand_dims(logsumexp(x,axis),axis)

def sigmoid(x):
    #Numerically stable sigmoid
    return np.exp(log_sigmoid(x))

def log_sigmoid(x):
    m = np.maximum(-x,0)
    return -(np.log(np.exp(-m) + np.exp(-x-m)) + m)

def logsumexp(x,axis=0):
    m = x.max(axis)
    if (len(x.shape) > 1):
        return np.log(np.sum(np.exp(x-np.expand_dims(m,axis)),axis)) + m
    else:
        return np.log(np.sum(np.exp(x-m))) + m