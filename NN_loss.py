import torch
import torch.nn as nn
from utils import network_params

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
    def __init__(self, criterion, flat_params, sigma_posterior_, model, d_size):
        
        super(mnnLoss, self).__init__()
        self.sigma_posterior_ = sigma_posterior_
        self.flat_params = flat_params
        self.d_size = d_size
        self.model = model
        self.criterion = criterion
    
    def forward(self, images, labels):
        self.noise = torch.randn(self.d_size) * torch.exp(self.sigma_posterior_)
        modified_parameters = self.flat_params +  self.noise 
        indi = 0
        for name,ind,shape_ in network_params(self.model):
            self.model.state_dict()[name].data.copy_(modified_parameters[indi:indi+ind].view(shape_)) 
            indi = ind
            
        outputs = self.model(images)
        loss = self.criterion(outputs.float(), labels.long())
        
        return loss

   

