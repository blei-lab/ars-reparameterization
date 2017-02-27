import numpy as np
import numpy.random as npr
import scipy.special as sp

# Score function estimator
from dirichlet_score import *

def grad_entropy(alpha):
    K = alpha.shape[0]
    alphaSum = np.sum(alpha)
    
    return -(K-alphaSum)*sp.polygamma(1,alphaSum) - (alpha-1.)*sp.polygamma(1,alpha)

# Define score function estimator (based on samples lmbda from G(alpha,beta))
def scoreEstimator(alpha,alpha0,x,S=100):
    """
    Form score function estimator based on samples lmbda.
    """
    gradient = np.zeros(alpha.shape[0])
    p = npr.dirichlet(alpha,size=S)
    lQ = logQ(p,alpha)
    lP = log_prior(p,alpha0)
    lL = log_likelihood(p,x)
    gradLQ = grad_logQ(p,alpha)
    
    temp = lP + lL - lQ
    for k in range(alpha.shape[0]):
        gradient[k] = np.mean(gradLQ[:,k]*temp)
    
    return gradient

def gamma_h(epsilon, alpha):
    """
    Reparameterization for gamma rejection sampler without shape augmentation.
    """
    b = alpha - 1./3.
    c = 1./np.sqrt(9.*b)
    v = 1.+epsilon*c
    
    return b*(v**3)    

def gamma_grad_h(epsilon, alpha):
    """
    Gradient of reparameterization without shape augmentation.
    """
    b = alpha - 1./3.
    c = 1./np.sqrt(9.*b)
    v = 1.+epsilon*c
    
    return v**3 - 13.5*epsilon*b*(v**2)*(c**3)

def gamma_h_boosted(epsilon, u, alpha):
    """
    Reparameterization for gamma rejection sampler with shape augmentation.
    """
    B = u.shape[1]
    K = alpha.shape[0]
    alpha_vec = np.tile(alpha,(B,1)).T + np.tile(np.arange(B),(K,1))
    u_pow = np.power(u,1./alpha_vec)
    
    return np.prod(u_pow,axis=1) * gamma_h(epsilon, alpha+B)
    
def gamma_grad_h_boosted(epsilon, u, alpha):
    """
    Gradient of reparameterization with shape augmentation.
    """
    B = u.shape[1]
    K = alpha.shape[0]
    h_val = gamma_h(epsilon, alpha+B)
    h_der = gamma_grad_h(epsilon, alpha+B)
    alpha_vec = np.tile(alpha,(B,1)).T + np.tile(np.arange(B),(K,1))
    u_pow = np.power(u,1./alpha_vec)
    u_der = -np.log(u)/alpha_vec**2
    
    return np.prod(u_pow,axis=1) * h_val * (h_der/h_val + np.sum(u_der,axis=1))
    
def gamma_grad_logr(epsilon, alpha):
    """
    Gradient of log-proposal.
    """
    b = alpha - 1./3.
    c = 1./np.sqrt(9.*b)
    v = 1.+epsilon*c
    
    return -0.5/b + 9.*epsilon*(c**3)/v
    
def gamma_grad_logq(epsilon, alpha):
    """
    Gradient of log-Gamma at proposed value.
    """
    h_val = gamma_h(epsilon, alpha)
    h_der = gamma_grad_h(epsilon, alpha)
    
    return np.log(h_val) + (alpha-1.)*h_der/h_val - h_der - sp.digamma(alpha)

def gamma_correction(epsilon, alpha):
    """
    Correction term grad (log q - log r)
    """
    return gamma_grad_logq(epsilon, alpha) - gamma_grad_logr(epsilon,alpha)
    
def grad_logp(p, alpha0, x):
    """
    d log p / dz
    """
    return (alpha0-1.+x)/p

def logp(p,alpha0,x):
    """
    Log-joint, log p
    """
    aSum = np.sum(alpha0)
    K = alpha0.shape[0]
    
    normConst = sp.gammaln(aSum)+sp.gammaln(1+np.sum(x))-np.sum(sp.gammaln(alpha0))
    normConst -= sp.gammaln(x+1)
    
    return np.sum((alpha0-1.)*np.log(p)) + np.sum(x*np.log(p)) + normConst

def calc_epsilon(p, alpha):
    """
    Calculate the epsilon accepted by Numpy's internal Marsaglia & Tsang
    rejection sampler. (h is invertible)
    """
    sqrtAlpha = np.sqrt(9.*alpha-3.)
    powZA = np.power(p/(alpha-1./3.),1./3.)
    
    return sqrtAlpha*(powZA-1.)
    
# Reparameterization gradient
def reparam_gradient(alpha,alpha0,x,corr=True,B=0):
    K = alpha.shape[0]
    gradient = np.zeros(K)
    jacob = np.zeros((K,K))
    if B == 0:
        p = npr.gamma(alpha,1.)
        epsilon = calc_epsilon(p, alpha)
        p /= np.sum(p)
        h_val = gamma_h(epsilon, alpha)
        h_der = gamma_grad_h(epsilon, alpha)
        h_sum = np.sum(h_val)
        for k in range(K):
            jacob[k,:] = -h_val
            jacob[k,k] += h_sum
            jacob[k,:] *= h_der[k]
        jacob /= h_sum**2
        logp_der = grad_logp(p, alpha0, x)
        gradient = np.dot(jacob,logp_der) + grad_entropy(alpha)
        if corr:
            gradient += logp(p,alpha0,x)*gamma_correction(epsilon, alpha)
    else:
        gam = npr.gamma(alpha+B,1.)
        u = npr.rand(K,B)
        epsilon = calc_epsilon(gam, alpha+B)
        h_val = gamma_h_boosted(epsilon,u,alpha)
        h_der = gamma_grad_h_boosted(epsilon,u,alpha)

        h_sum = np.sum(h_val)
        for k in range(K):
            jacob[k,:] = -h_val
            jacob[k,k] += h_sum
            jacob[k,:] *= h_der[k]
        jacob /= h_sum**2
        p = h_val
        p /= np.sum(p)
        logp_der = grad_logp(p, alpha0, x)
        gradient = np.dot(jacob,logp_der) + grad_entropy(alpha)
        if corr:
            gradient += logp(p,alpha0,x)*gamma_correction(epsilon, alpha+B)
            
    return gradient

