import torch.nn as nn


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

        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc3 = nn.Linear(hidden_size, num_classes, bias=True)
        
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

        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc3 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc4 = nn.Linear(hidden_size, num_classes, bias=True)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


class FeedForwardNeuralNet3R(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(torch.nn.Linear(input_size, hidden_size, bias=True),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(hidden_size, hidden_size, bias=True),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(hidden_size, hidden_size, bias=True),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(hidden_size, num_classes, bias=True),
                                    )
    def forward(self, x):
        out = self.main(x)
        return out

def weights_init(m):
    """
    Function for weights initializing . According to the paper :
    First layer bias (weights initialized rondomly and bias to 0.1)
    Remaining layers (weights initialized rondomly and bias to 0)
    
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.04)
        m.bias.data.fill_(0)
        
def create_network(nb_layers, input_size, hidden_size, num_classes):
    if (nb_layers == 1):
        network = FeedForwardNeuralNet(input_size, hidden_size, num_classes)
    elif (nb_layers == 2):
        network = FeedForwardNeuralNet2(input_size, hidden_size, num_classes)
    elif (nb_layers == 3):
        network = FeedForwardNeuralNet3(input_size, hidden_size, num_classes)
    else :
        raise Exception('The number of layers should not exceed 3')

    network.apply(weights_init)
    network.fc1.bias.data.fill_(0.1)
    
    return network