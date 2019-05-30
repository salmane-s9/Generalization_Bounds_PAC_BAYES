import torch
import torch.nn as nn
from utils import network_params
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class mnnLoss(nn.Module):
    """ class for calcuting surrogate loss of the SNN (first term in minimization problem).
    Parameters
    ----------
    flat_params : torch array of shape (d_size,)
        flat array of NN parameters
    sigma_posterior_ : {torch array, Parameter}
        Posterior distribution N(w,s) variance .
    model : nn.Module 
        Architecture of neural network to evaluate
    d_size : int
        Number of NN parameters  
    """
    def __init__(self, criterion, flat_params, sigma_posterior_, model, d_size, device):
        
        super(mnnLoss, self).__init__()
        self.sigma_posterior_ = nn.Parameter(sigma_posterior_).to(device)
        self.flat_params = nn.Parameter(flat_params).to(device)
        self.d_size = d_size
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.device = device
    
    def forward(self, images, labels):
        self.noise = torch.randn(self.d_size).to(self.device) * torch.exp(self.sigma_posterior_)
        vector_to_parameters(self.flat_params + self.noise, self.model.parameters())
        outputs = self.model(images)
        # loss = self.criterion(outputs.float(), labels.long())
        loss = F.cross_entropy(outputs.float(), labels.long())
        return loss

   

