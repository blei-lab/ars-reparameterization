from autograd import grad
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.special as sp

from gamma_def import *

# Define helper functions for score fnc estimator
def logQ(sample, alpha, m):
    """
    Evaluates log of variational approximation, vectorized.
    """
    temp = alpha*(np.log(alpha)-np.log(m))
    temp += (alpha-1.)*np.log(sample)
    temp -= alpha*sample/m
    temp -= np.log(sp.gamma(alpha))
    return temp

def grad_logQ(sample,alpha,m):
    """
    Evaluates the gradient of the log of variational approximation, vectorized.
    """
    gradient = np.zeros((alpha.shape[0],2))

    gradient[:,0] = np.log(alpha) - np.log(m) + 1. + np.log(sample) - sample/m
    gradient[:,0] -= sp.digamma(alpha)
    gradient[:,1] = -alpha/m + alpha*sample/m**2
    
    return gradient

# Define score function estimator
def score_estimator(alpha,m,x,K,alphaz,S=100):
    """
    Form score function estimator based on samples lmbda.
    """
    N = x.shape[0]
    if x.ndim == 1:
        D = 1
    else:
        D = x.shape[1]
    num_z = N*np.sum(K)
    L = K.shape[0]
    gradient = np.zeros((alpha.shape[0],2))
    f = np.zeros((2*S,alpha.shape[0],2))
    h = np.zeros((2*S,alpha.shape[0],2))
    for s in range(2*S):
        lmbda = npr.gamma(alpha,1.)
        lmbda[lmbda < 1e-300] = 1e-300
        zw = m*lmbda/alpha
        lQ = logQ(zw,alpha,m)
        gradLQ = grad_logQ(zw,alpha,m)
    
        lP = logp(zw, K, x, alphaz)
        temp = lP - np.sum(lQ)
        f[s,:,:] = temp*gradLQ
        
        h[s,:,:] = gradLQ
        
    # CV
    covFH = np.zeros((alpha.shape[0],2))
    covFH[:,0] =  np.diagonal(np.cov(f[S:,:,0],h[S:,:,0],rowvar=False)[:alpha.shape[0],alpha.shape[0]:])
    covFH[:,1] =  np.diagonal(np.cov(f[S:,:,1],h[S:,:,1],rowvar=False)[:alpha.shape[0],alpha.shape[0]:])
    a = covFH / np.var(h[S:,:,:],axis=0)    
    
    return np.mean(f[:S,:,:],axis=0) - a*np.mean(h[:S,:,:],axis=0)
