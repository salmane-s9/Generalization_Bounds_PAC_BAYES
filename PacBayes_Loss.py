
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from utils import calc_BRE_term,calc_kullback_leibler



# In[ ]:


class PacBayesLoss(nn.Module):
    
    def __init__(self, lambda_prior_ ,sigma_posterior_ , params, conf_param , Precision , 
                 bound , data_size ,reduce=None, reduction='mean'):
        
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

