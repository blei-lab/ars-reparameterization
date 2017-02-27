import numpy as np
import numpy.random as npr
import scipy.special as sp

# Define helper functions for score fnc estimator
def logQ(p, alpha):
    """
    Evaluates log of variational approximation, vectorized.
    """
    temp = np.zeros_like(p)
    normconst = np.log(sp.gamma(np.sum(alpha)))-np.sum(np.log(sp.gamma(alpha)))
    
    for i in range(temp.shape[0]):
        temp[i,:] = (alpha-1.)*np.log(p[i,:])
        
    return np.sum(temp,axis=1) + normconst

def grad_logQ(p,alpha):
    """
    Evaluates the gradient of the log of variational approximation, vectorized.
    """
    const = sp.digamma(np.sum(alpha))
    gradient = np.zeros((p.shape[0],alpha.shape[0]))
    for k in range(alpha.shape[0]):
        gradient[:,k] = np.log(p[:,k])-sp.digamma(alpha[k])
        
    return gradient + const

def log_prior(p,alpha0):
    """
    Evaluates log of the prior, vectorized.
    """
    temp = np.zeros_like(p)
    normconst = np.log(sp.gamma(np.sum(alpha0)))-np.sum(np.log(sp.gamma(alpha0)))
    
    for i in range(temp.shape[0]):
        temp[i,:] = (alpha0-1.)*np.log(p[i,:])
        
    return np.sum(temp,axis=1) + normconst

def log_likelihood(p,x):
    """
    Evaluates log of the likelihood.
    """
    N = np.sum(x)
    temp = np.zeros_like(p)
    normconst = np.log(sp.gamma(N+1))-np.sum(np.log(sp.gamma(x+1)))

    for i in range(temp.shape[0]):
        temp[i,:] = x*np.log(p[i,:])
        
    return np.sum(temp,axis=1) + normconst
