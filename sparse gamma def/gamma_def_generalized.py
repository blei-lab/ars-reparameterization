from autograd import grad
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.special as sp

from gamma_def import *

def fun_Tinv(z,alpha):
    return (np.log(z)-sp.digamma(alpha))/np.sqrt(sp.polygamma(1,alpha))

def fun_T(eps,alpha):
    return np.exp(eps*np.sqrt(sp.polygamma(1,alpha))+sp.digamma(alpha))
    
def fun_H(eps,alpha):
    poly1 = sp.polygamma(1,alpha)
    poly2 = sp.polygamma(2,alpha)
    T_val = fun_T(eps,alpha)
    
    return T_val*(0.5*eps*poly2/np.sqrt(poly1) + poly1)

def fun_U(eps,alpha):
    poly1 = sp.polygamma(1,alpha)
    poly2 = sp.polygamma(2,alpha)
 
    return 0.5*poly2/poly1 + 0.5*poly2*eps/np.sqrt(poly1) + poly1
    
def grad_logQ_Z(samp,alpha):
    return (alpha-1.)/samp - 1.

def grad_logQ_alpha(samp,alpha):
    return np.log(samp)-sp.digamma(alpha)
    
# Generalized gradient
def grep_gradient(alpha,m,x,K,alphaz):
    gradient = np.zeros((alpha.shape[0],2))
    lmbda = npr.gamma(alpha,1.)
    lmbda[lmbda<1e-5] = 1e-5
    Tinv_val = fun_Tinv(lmbda,alpha)
    h_val = fun_H(Tinv_val,alpha)
    u_val = fun_U(Tinv_val,alpha)
    
    zw = m*lmbda/alpha
    zw[zw < 1e-5] = 1e-5
    logp_der = grad_logp(zw, K, x, alphaz)
    logp_val = logp(zw, K, x, alphaz)
    logq_der = grad_logQ_Z(zw,alpha)
    
    gradient[:,0] = logp_der*(h_val-lmbda/alpha)*m/alpha
    gradient[:,1] = logp_der*lmbda/alpha
    gradient[:,0] += logp_val*(np.log(lmbda)+(alpha/lmbda-1.)*h_val-sp.digamma(alpha)+sp.polygamma(2,alpha)/2./sp.polygamma(1,alpha))
    gradient += grad_entropy(alpha,m)
   
    return gradient

