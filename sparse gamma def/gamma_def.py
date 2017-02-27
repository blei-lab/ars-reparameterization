from autograd import grad
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.special as sp

w_shp = 0.1
w_rte = 0.3
z_shp = 0.1
z_rte = 0.1

def entropy(alpha,m):
    return alpha+np.log(m)-np.log(alpha)+sp.gammaln(alpha)+(1.-alpha)*sp.digamma(alpha)
    
def grad_entropy(alpha, m):
    gradient = np.zeros((alpha.shape[0],2))
    gradient[:,0] = 1. - 1./alpha + (1.-alpha)*sp.polygamma(1,alpha)
    gradient[:,1] = 1./m

    return gradient

#def logp(zw,K,x,alphaz):
    #assert isinstance(K,np.ndarray),"K is assumed to be a Numpy array"
    #N = x.shape[0]
    #D = x.shape[1]

    #L = K.shape[0]
    #num_z = N*np.sum(K)
    #z = zw[:num_z]
    #w = zw[num_z:]
    #num_w = w.shape[0]
    
    #norm_const = N*K[-1]*(z_shp*np.log(z_rte) + np.log(sp.gamma(z_shp))) - np.sum(sp.gammaln(x+1))
    #norm_const -= (num_z-N*K[-1])*sp.gammaln(alphaz)
    #norm_const += num_w*(w_shp*np.log(w_rte) - sp.gammaln(w_shp))
    
    ## Likelihood
    #z1_sum_w0 = np.dot(z[:N*K[0]].reshape((N,K[0])),w[:D*K[0]].reshape((K[0],D)))
    #log_likelihood = np.sum(x*np.log(z1_sum_w0))-np.sum(z1_sum_w0)
    #log_w = (w_shp-1.)*np.sum(np.log(w))-w_rte*np.sum(w)
    
    #ind_w = D*K[0]
    #ind_z = N*K[0]
    #log_z = (z_shp-1.)*np.sum(np.log(z[num_z-N*K[-1]:]))-z_rte*np.sum(z[num_z-N*K[-1]:])
    
    #for ell in range(L-1):
        #zl1_sum_wl = np.dot(z[ind_z:ind_z+N*K[ell+1]].reshape((N,K[ell+1])), w[ind_w:ind_w+K[ell]*K[ell+1]].reshape((K[ell+1],K[ell])))
        #zl = z[ind_z-N*K[ell]:ind_z].reshape((N,K[ell]))
        #log_z += np.sum(alphaz*(np.log(alphaz)-np.log(zl1_sum_wl)))
        #log_z += np.sum((alphaz-1.)*np.log(zl))
        #log_z += -alphaz*np.sum(zl/zl1_sum_wl)
        #ind_z += N*K[ell+1]
        #ind_w += K[ell]*K[ell+1]
        
    #return log_likelihood + log_z + log_w + norm_const

def logp(zw,K,x,alphaz):
    assert isinstance(K,np.ndarray),"K is assumed to be a Numpy array"
    N = x.shape[0]
    D = x.shape[1]

    L = K.shape[0]
    num_z = N*np.sum(K)
    z = zw[:num_z]
    w = zw[num_z:]
    num_w = w.shape[0]
    
    log_prior = 0.
    log_likelihood = 0.
    
    # Prior for weights
    log_prior += np.sum(w_shp*np.log(w_rte)+(w_shp-1.)*np.log(w)-w_rte*w-sp.gammaln(w_shp))
    
    # Prior for top layer
    log_prior += np.sum(z_shp*np.log(z_rte)+(z_shp-1.)*np.log(z[-N*K[-1]:])-z_rte*z[-N*K[-1]:]-sp.gammaln(z_shp))
    
    # Likelihood
    z1 = z[:N*K[0]].reshape((N,K[0]))
    w0 = w[:D*K[0]].reshape((K[0],D))
    z1_sum_w0 = np.dot(z1,w0)
    log_likelihood = np.sum(x*np.log(z1_sum_w0)-z1_sum_w0-sp.gammaln(x+1))
    
    if L > 1:
        # Layer 1
        z2 = z[N*K[0]:N*(K[0]+K[1])].reshape((N,K[1]))
        w1 = w[D*K[0]:D*K[0]+K[0]*K[1]].reshape((K[1],K[0]))
        z2_sum_w1 = np.dot(z2,w1)
        aux = alphaz/z2_sum_w1
        log_prior += np.sum(alphaz*np.log(aux)+(alphaz-1.)*np.log(z1)-aux*z1-sp.gammaln(alphaz))
        
        # Layer 2
        z3 = z[N*(K[0]+K[1]):N*(K[0]+K[1]+K[2])].reshape((N,K[2]))
        w2 = w[D*K[0]+K[0]*K[1]:D*K[0]+K[0]*K[1]+K[1]*K[2]].reshape((K[2],K[1]))
        z3_sum_w2 = np.dot(z3,w2)
        aux = alphaz/z3_sum_w2
        log_prior += np.sum(alphaz*np.log(aux)+(alphaz-1.)*np.log(z2)-aux*z2-sp.gammaln(alphaz))
    
    return log_prior+log_likelihood
    
grad_logp = grad(logp,argnum=0)

def estimate_elbo(alpha,m,K,x,alphaz,S=1):
    elbo = np.zeros(S)
    for s in range(S):
        zw = npr.gamma(alpha,1.)*m/alpha
        zw[zw < 1e-300] = 1e-300
        elbo[s] = logp(zw,K,x,alphaz)
        
    return np.mean(elbo) + np.sum(entropy(alpha,m))
