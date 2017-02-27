from autograd import grad
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.special as sp

from gamma_def import *

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

def calc_epsilon(p, alpha):
    """
    Calculate the epsilon accepted by Numpy's internal Marsaglia & Tsang
    rejection sampler. (h is invertible)
    """
    sqrtAlpha = np.sqrt(9.*alpha-3.)
    powZA = np.power(p/(alpha-1./3.),1./3.)
    
    return sqrtAlpha*(powZA-1.)
    
# Reparameterization gradient
def reparam_gradient(alpha,m,x,K,alphaz,corr=True,B=0):
    gradient = np.zeros((alpha.shape[0],2))
    if B == 0:
        assert np.min(alpha)>= 1.,"Needs shape augmentation"
        lmbda = npr.gamma(alpha,1.)
        lmbda[lmbda < 1e-300] = 1e-300
        zw = m*lmbda/alpha
        epsilon = calc_epsilon(lmbda, alpha)
        h_val = gamma_h(epsilon, alpha)
        h_der = gamma_grad_h(epsilon, alpha)
        logp_der = grad_logp(zw, K, x, alphaz)
        gradient[:,0] = logp_der*m*(alpha*h_der-h_val)/alpha**2
        gradient[:,1] = logp_der*h_val/alpha
        gradient += grad_entropy(alpha,m)
        if corr:
            gradient[:,0] += logp(zw, K, x, alphaz)*gamma_correction(epsilon, alpha)
    else:
        lmbda = npr.gamma(alpha+B,1.)
        lmbda[lmbda < 1e-5] = 1e-5
        u = npr.rand(alpha.shape[0],B)
        epsilon = calc_epsilon(lmbda, alpha+B)
        h_val = gamma_h_boosted(epsilon,u,alpha)
        h_der = gamma_grad_h_boosted(epsilon,u,alpha)
        zw = h_val*m/alpha
        zw[zw < 1e-5] = 1e-5
        logp_der = grad_logp(zw, K, x, alphaz)
        gradient[:,0] = logp_der*m*(alpha*h_der-h_val)/alpha**2
        gradient[:,1] = logp_der*h_val/alpha
        gradient += grad_entropy(alpha,m)
        if corr:
            gradient[:,0] += logp(zw, K, x, alphaz)*gamma_correction(epsilon, alpha+B)
    return gradient

