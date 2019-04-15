import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from utils import *
from math import log
import copy
import time

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
                 bound, data_size, initial_weights, device):
        
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
        if (initial_weights[0]=="random"):
            self.params_0 = torch.randn(self.flat_params.size()).to(self.device)
        elif (initial_weights[0]=="zeros"):
            self.params_0 = torch.zeros(self.flat_params.size()).to(self.device)
        else:
            self.params_0 = initial_weights[1]
        self.d_size = flat_params.size()[0]
        self.kl_value = None
        for p in self.model.parameters():
                p.requires_grad = False
        
    def forward(self):
        Bre_loss, kl_value = calc_BRE_term(self.precision, self.conf_param, self.bound, self.flat_params, 
                                 self.params_0, self.lambda_prior_, self.sigma_posterior_, 
                                 self.data_size, self.d_size)
        self.kl_value = kl_value
        return Bre_loss
    
    def compute_bound(self, train_loader, delta_prime, n_mtcarlo_approx):
        """
         Returns:
            SNN_train_error : upper bound on the train error of the Stochastic neural network by application of Theorem of
                              the sample convergence bound
            final_bound : Final Pac Bayes bound by application of Paper theorem on SNN_train_error 
        """

        snn_error = self.SNN_error(train_loader, delta_prime, n_mtcarlo_approx) 
        final_bound = []

        j_round = torch.round(self.precision * (log(self.bound) - (2 * self.lambda_prior_)))
        lambda_prior_ = 0.5 * (log(self.bound)- (j_round/self.precision)).clone().detach()

        Bre_loss, kl = calc_BRE_term(self.precision, self.conf_param, self.bound, self.flat_params, 
                        self.params_0, lambda_prior_, self.sigma_posterior_, 
                        self.data_size, self.d_size)
        
        if torch.cuda.is_available():
            cuda_tensor = Bre_loss.cuda()
            Bre_loss = cuda_tensor.cpu().detach().numpy()
        else:
            Bre_loss = Bre_loss.detach().numpy()

        for i in snn_error:          
            final_bound.append(solve_kl_sup(i, 2 * (Bre_loss**2)))
        
        return snn_error, final_bound, kl
    
    def sample_weights(self):      
        """
       Sample weights from the posterior distribution Q(flat_params, Sigma_posterior)
        """
        return self.flat_params + torch.randn(self.d_size).to(self.device) * torch.exp(self.sigma_posterior_)
    
    def SNN_error(self, loader, delta_prime, n_mtcarlo_approx):
        """
      Compute upper bound on the error of the Stochastic neural network by application of Theorem of the sample convergence bound 
        """
        samples_errors = 0.
        net_params = network_params(self.model)
        snn_error = []
        
        with torch.no_grad():
            t = time.time()
            iter_counter = 1000
            for i in range(n_mtcarlo_approx):
                nn_model = apply_weights(self.model, self.sample_weights(), net_params)
                samples_errors += test_error(loader, nn_model, self.device)
                if i == iter_counter:
                    snn_error_intermed = solve_kl_sup(samples_errors/i, (log(2/delta_prime)/i))
                    snn_error.append(snn_error_intermed)
                    print("It's {}th Monte-Carlo iteration".format(i))
                    print("Computational time for {} is {}".format(i, time.time() - t))
                    iter_counter += 1000

        snn_final_error = solve_kl_sup(samples_errors/n_mtcarlo_approx, (log(2/delta_prime)/n_mtcarlo_approx))
        snn_error.append(snn_final_error)
        
        return snn_error

