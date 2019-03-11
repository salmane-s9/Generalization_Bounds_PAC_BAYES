import torch
import torch.nn as nn
from math import log,pi
import numpy as np
from scipy import optimize

def calc_kullback_leibler(lambda_prior ,sigma_post ,params , params_0 , d_size):
    """
    explicit calculation of KL divergence between prior N(0,lambda_prior * Id) and posterior N(flat_params, sigma_posterior_)
    """
    
    tr = torch.norm(sigma_post, p=1)/ lambda_prior
    
    l2 = torch.pow(torch.norm(params -params_0, p=2), 2)/ lambda_prior
    d = d_size 

    logdet_prior = d * torch.log(lambda_prior)
    
    logdet_post = torch.sum(torch.log(sigma_post))

    kl = (tr + l2 - d + logdet_prior - logdet_post ) / 2.

    return kl

def calc_BRE_term(Precision ,conf_param ,bound ,params , params_0,lambda_prior_ ,sigma_posterior_,data_size,d_size): 
    """
   Explicit Calculation of the second term of the optimization problem (BRE)
    """
 
    lambda_prior = torch.clamp(torch.exp(2 * lambda_prior_ ), min = 1e-38, max = bound - 1e-8)
    sigma_post = torch.exp(2 * sigma_posterior_)
    
    kl = calc_kullback_leibler(lambda_prior, sigma_post ,params , params_0 , d_size)
    
    log_log = 2* torch.log(Precision* (torch.log(bound /lambda_prior)))

    m = data_size
    log_ = log((((pi**2) * m)/(6* conf_param)))

    bre = torch.sqrt((kl + log_log + log_) / (2 *(m-1)))

    return bre

def network_params(model):
    """
    Return a list containing names and shapes of neural network layers
    """

    layers = []
    ind = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            shape = model.state_dict()[name].shape
            params = np.ravel(param.data.numpy())
            ind2 = np.size(params)
            ind = ind2
            layers.append((name,ind,shape))
        
    return layers
    
def load_train_weights(model ,weights):
    """
    Load trained weights into a neural network model
    """
    pretrained_dict = torch.load(weights)
    model_dict = model.state_dict()

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.state_dict()
    return model

def test_error(loader, nn_model, device):
    """
    Compute the empirical error of neural network on a dataset loader
    """
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            outputs = nn_model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()
        error = 1. - correct / total
        return(error)
    
def apply_weights(model, modified_parameters):
    """
    Modify the parameters of a neural network 
    """
    indi = 0
    for name,ind,shape_ in network_params(model):
        model.state_dict()[name].data.copy_(modified_parameters[indi:indi+ind].view(shape_)) 
        indi += ind
    return(model)


def KL(Q, P):
    """
    Compute Kullback-Leibler (KL) divergence between distributions Q and P.
    """
    return sum([ q*log(q/p) if q > 0. else 0. for q,p in zip(Q,P) ])


def KL_binomial(q, p):
    """
    Compute the KL-divergence between two Bernoulli distributions of probability
    of success q and p. That is, Q=(q,1-q), P=(p,1-p).
    """
    return KL([q, 1.-q], [p, 1.-p])

def solve_kl_sup(q, right_hand_side):
    """
    find x such that:
        kl( q || x ) = right_hand_side
        x > q
    """
    f = lambda x: KL_binomial(q, x) - right_hand_side

    if f(1.0-1e-9) <= 0.0:
        return 1.0-1e-9
    else:
        return optimize.brentq(f, q, 1.0-1e-9)
     
