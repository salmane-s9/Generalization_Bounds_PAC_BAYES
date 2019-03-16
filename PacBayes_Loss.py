import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from utils import *
# from utils import calc_BRE_term, calc_kullback_leibler, apply_weights, test_error, solve_kl_sup
from math import log
import copy


class PacBayesLoss(nn.Module):
    """ class for BRE loss (second term in minimization problem).
    Parameters
    ----------
    lambda_prior_ : int Parameter
        Prior distribution (P(0,λI) variance .
        
    sigma_posterior_ : {torch array, Parameter}
        Posterior distribution N(w,s) variance .
        
    net : Neural network model
        Feed Forward model .
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
    def __init__(self, lambda_prior_, sigma_posterior_, net, flat_params, conf_param, Precision, 
                 bound, data_size, device):
        
        super(PacBayesLoss, self).__init__()
        self.device = device
        self.model = copy.deepcopy(net).to(self.device)
        self.lambda_prior_ = nn.Parameter(lambda_prior_)
        self.sigma_posterior_ = nn.Parameter(sigma_posterior_)
        self.flat_params = nn.Parameter(flat_params)
        self.precision = Precision
        self.conf_param = conf_param
        self.bound = bound
        self.data_size = data_size
        self.params_0 = torch.randn(self.flat_params.size()).to(self.device)
        self.d_size = flat_params.size()[0]
        for p in self.model.parameters():
                p.requires_grad = False
        
    def forward(self):
        Bre_loss = calc_BRE_term(self.precision, self.conf_param, self.bound, self.flat_params, 
                                 self.params_0, self.lambda_prior_, self.sigma_posterior_, 
                                 self.data_size, self.d_size)
        return Bre_loss
    

    def compute_bound(self, train_loader, delta_prime, n_mtcarlo_approx):
        """
         Returns:
            SNN_train_error : upper bound on the train error of the Stochastic neural network by application of Theorem of
                              the sample convergence bound
            final_bound : Final Pac Bayes bound by application of Paper theorem on SNN_train_error 
        """
        SNN_train_error = self.SNN_error(train_loader, delta_prime, n_mtcarlo_approx, self.device) 
        
        j_round = torch.round(self.precision * (log(self.bound) - (2 * self.lambda_prior_)))
        lambda_prior_ = 0.5 * (log(self.bound)- (j_round/self.precision)).clone().detach()

        Bre_loss = calc_BRE_term(self.precision, self.conf_param, self.bound, self.flat_params, 
                                 self.params_0, lambda_prior_, self.sigma_posterior_, 
                                 self.data_size, self.d_size)
        
        final_bound = solve_kl_sup(SNN_train_error, Bre_loss)
        
        return SNN_train_error, final_bound
    
    def sample_weights(self):      
        """
       Sample weights from the posterior distribution Q(flat_params, Sigma_posterior)
        """
        return self.flat_params + torch.randn(self.d_size) * torch.exp(self.sigma_posterior_)
    
    def SNN_error(self, loader, delta_prime, n_mtcarlo_approx):
        """
      Compute upper bound on the error of the Stochastic neural network by application of Theorem of the sample convergence bound 
        """
        samples_errors = 0.
        net_params = network_params(self.model)
        
        for i in range(n_mtcarlo_approx):
            nn_model = apply_weights(self.model, self.sample_weights(), net_params)
            samples_errors += test_error(loader, nn_model, self.device)

        SNN_error = solve_kl_sup(samples_errors/n_mtcarlo_approx, (log(2/delta_prime)/n_mtcarlo_approx))
        
        return SNN_error

