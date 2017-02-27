import argparse
from autograd import grad
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.special as sp

from gamma_def import *
from gamma_def_rejection import *
from gamma_def_score import *

parser = argparse.ArgumentParser(description='Run Sparse gamma DEF example.')
parser.add_argument('--eta', type=float, help='Stepsize parameter')
parser.add_argument('-B', type=int, help='Shape Augmentation')
args = parser.parse_args()

eta = args.eta
B = args.B
if B > 1:
    correction = False
else:
    correction = True
    
n_iter = 10

# Setup sizes
K = np.array([100,40,15])
D = 64*64
N = 320
n_latent = N*np.sum(K)+K[0]*D+np.sum(K[1:]*K[:-1])
sigma = 0.1

# Load data
x = np.loadtxt('data/faces_training.csv',delimiter=',')
alphaz = 0.1

# Define truncation functions and stepsize updates
trunc_shape = np.log(np.exp(1e-3)-1.)
trunc_mean = np.log(np.exp(1e-4)-1.)

def stepSize(iteration,sPrev,gradient,eta=1.0):
    sCur = 0.1*(gradient**2) + 0.9*sPrev
    step = eta*np.power(iteration,-0.5+1e-16)/(1.+np.sqrt(sCur))
    
    return step,sCur

def truncate_params(params):
    ind = params[:,0] < trunc_shape
    params[ind,0] = trunc_shape
    ind = params[:,1] < trunc_mean
    params[ind,1] = trunc_mean
    return params

# Initialize
num_seed = 123
npr.seed(num_seed)
params_R = np.zeros((n_latent,2))
steps = np.ones((n_latent,2))
sCur_R = np.zeros((n_latent,2))
ELBO_R = np.zeros(n_iter)

params_R[:,0] = 0.5+sigma*npr.normal(size=n_latent)
params_R[:,1] = sigma*npr.normal(size=n_latent)

transformVar = np.log(1.+np.exp(params_R))
ELBO_R[0] = estimate_elbo(transformVar[:,0],transformVar[:,1],K,x,alphaz)

for n in range(1,n_iter):
    sGrad = reparam_gradient(transformVar[:,0],transformVar[:,1],x,K,alphaz,corr=correction,B=B)/(1.+np.exp(-params_R))
    steps,sCur_R = stepSize(n+1,sCur_R,sGrad,eta)
    
    params_R  = truncate_params(params_R+steps*sGrad)
    transformVar = np.log(1.+np.exp(params_R))
    ELBO_R[n] = estimate_elbo(transformVar[:,0],transformVar[:,1],K,x,alphaz)
    if np.mod(n,100) == 0:
        filename = 'results/Olivette_Eta'+str(eta)+'_B'+str(B)+'_corr'+str(correction)+'_ELBO.npy'
        np.save(filename, ELBO_R[:n_iter])
        filename = 'results/Olivette_Eta'+str(eta)+'_B'+str(B)+'_corr'+str(correction)+'_K1_'+str(K[0])+'_K2_'+str(K[1])+'_K3_'+str(K[2])+'_params_R.npy'
        np.save(filename,np.log(1.+np.exp(params_R)))

filename = 'results/Olivette_Eta'+str(eta)+'_B'+str(B)+'_corr'+str(correction)+'_ELBO.npy'
np.save(filename, ELBO_R[:n_iter])
filename = 'results/Olivette_Eta'+str(eta)+'_B'+str(B)+'_corr'+str(correction)+'_K1_'+str(K[0])+'_K2_'+str(K[1])+'_K3_'+str(K[2])+'_params_R.npy'
np.save(filename,np.log(1.+np.exp(params_R)))
