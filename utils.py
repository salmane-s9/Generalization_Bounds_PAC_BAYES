import torch
import torch.nn as nn
from math import log,pi
import numpy as np



def calc_kullback_leibler(lambda_prior ,sigma_post ,params , params_0 , d_size):
    # explicit calculation of KL divergence between prior N(0,lambda * Id) and posterior N(w, s)

#     assert (torch.is_tensor(lambda_prior) and torch.is_tensor(sigma_post) )
    tr = torch.norm(sigma_post , p=1)/ lambda_prior
    
    l2 = torch.norm(params -params_0 , p=2)/ lambda_prior
    d = d_size 

    logdet_prior = d * torch.log(lambda_prior)
    
    logdet_post = torch.sum(torch.log(sigma_post))

    kl = (tr + l2 - d + logdet_prior - logdet_post ) / 2.

    return kl

def calc_BRE_term(Precision ,conf_param ,bound ,params , params_0,lambda_prior_ ,sigma_posterior_,data_size,d_size): 
#   Explicit Calculation of the second term of the bound (BRE)

    lambda_prior = torch.exp(2 * lambda_prior_ )
    sigma_post = torch.exp(2 * sigma_posterior_ )
    
    kl = calc_kullback_leibler(lambda_prior, sigma_post ,params , params_0 , d_size)
    
    assert bound > lambda_prior
    log_log = 2* torch.log(Precision* torch.log(bound /lambda_prior))

    m = data_size
    log_ = log((((pi**2) * m)/(6* conf_param)))

    bre = torch.sqrt((kl + log_log + log_) / (2 *(m-1)))

    return bre

def network_params(model):
#       return network parameters and layers 
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


class FeedForwardNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size, bias =True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes, bias =True)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    
class FeedForwardNeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size, bias =True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias =True)
        self.fc3 = nn.Linear(hidden_size, num_classes, bias =True)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

    
class FeedForwardNeuralNet3(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size, bias =True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias =True)
        self.fc3 = nn.Linear(hidden_size, hidden_size, bias =True)
        self.fc4 = nn.Linear(hidden_size, num_classes, bias =True)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


# Another solution
class FeedForwardNeuralNet3R(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(torch.nn.Linear(input_size, hidden_size, bias =True),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(hidden_size, hidden_size, bias =True),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(hidden_size, hidden_size, bias =True),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(hidden_size, num_classes, bias =True),
                                    )
    def forward(self, x):
        out = self.main(x)
        return out


    
def load_train_weights(model ,weights):
    pretrained_dict = torch.load(weights)
    model_dict = model.state_dict()

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.state_dict()
    return model


def test_error(*, loader, nn_model, device):
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

        print('{0:.3f} '.format(1. - correct / total))
