import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from utils import calc_BRE_term,calc_kullback_leibler


class PacBayesLoss(nn.Module):
    """ class for BRE loss (second term in minimization problem).
    Parameters
    ----------
    lambda_prior_ : int Parameter
        Prior distribution (P(0,λI) variance .
        
    sigma_posterior_ : {torch array, Parameter}
        Posterior distribution N(w,s) variance .
        
    params : Neural network parameters
        Neural network parameters .
    conf_param : float 
        confidence parameter .
    Precision : int 
        precision parameter for lambda_prior .
    bound : float 
       upper bound for lambda_prior .
    data_size : int 
        size of training data .  
        
    Attributes
    ----------
    
    d_size : int
        Number of NN parameters
    flat_params : torch array of shape (d_size,)
        flat array of NN parameters
    params_0 : torch array of shape (d_size,)
        mean of Prior distribution .
    """
    def __init__(self, lambda_prior_ ,sigma_posterior_ , params, conf_param , Precision , 
                 bound , data_size):
        
        super(PacBayesLoss, self).__init__()
        self.lambda_prior_ = nn.Parameter(lambda_prior_)
        self.sigma_posterior_ = nn.Parameter(sigma_posterior_)
        self.params = params.parameters()
        self.flat_params = nn.Parameter(parameters_to_vector(params.parameters()))
        self.precision = Precision
        self.conf_param = conf_param
        self.bound = bound
        self.data_size = data_size
        self.params_0 = torch.randn(self.flat_params.size())
        self.d_size = sum(p.numel() for p in self.params if p.requires_grad)
       
        
    def forward(self):
        Bre_loss = calc_BRE_term(self.precision, self.conf_param, self.bound, self.flat_params, 
                                 self.params_0, self.lambda_prior_, self.sigma_posterior_, 
                                 self.data_size, self.d_size)
        
        
        return Bre_loss

