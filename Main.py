import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.autograd import Variable
from math import log , pi
from torch.nn.utils import parameters_to_vector
from PacBayes_Loss import PacBayesLoss
from NN_loss import mnnLoss
from utils import *
from Mnist_dataset import binary_mnist_loader
import matplotlib.pyplot as plt
from Architectures import * 


def main(test_cuda=False):
    
    print('-'*80)
    device = torch.device("cuda" if test_cuda else "cpu")
    INPUT_SIZE = 784
    HIDDEN_SIZE = [300 , 600 , 1200]
    NUM_CLASSES = 2
    
#     We will work with the model ['T-600']
    initial_net = FeedForwardNeuralNet(INPUT_SIZE, HIDDEN_SIZE[1], NUM_CLASSES)
    net = load_train_weights(initial_net,'SGD_solutions/T-600.ckpt')
    
    train_loader, test_loader = binary_mnist_loader()
    
    conf_param=0.025 
    Precision= 100 
    bound=0.1 
    data_size= 55000
    learning_rate = 0.001
    
    lambda_prior = torch.tensor(-3. ,device=device).requires_grad_()
    sigma_posterior = torch.abs(parameters_to_vector(net.parameters())).requires_grad_()
#     d = sum(p.numel() for p in net.parameters() if p.requires_grad)
#     sigma_posterior = torch.tensor([-3.]*d)
    BRE = PacBayesLoss(lambda_prior, sigma_posterior, net, conf_param, Precision, bound, 
                      data_size).to(device)
    
    optimizer = torch.optim.RMSprop(BRE.parameters(), lr= learning_rate, alpha=0.9)
    criterion  = nn.CrossEntropyLoss()
    nnloss = mnnLoss(criterion, BRE.flat_params, BRE.sigma_posterior_ , net , BRE.d_size)
    epochs = 2
#     norm_change = []
#     initial_weights = parameters_to_vector(net.parameters())
    for epoch in np.arange(1,epochs): 
        print(" \n Epoch {} :  ".format(epoch), end="\n")
        
        for i, (images, labels) in enumerate(train_loader):
                
#                 if i >20 : break
                print("\r{}%".format(100 * i // BRE.data_size), end="")
                
                images = images.reshape(-1, 28 * 28).to(device)
                labels = labels.to(device)
                
                loss1 = BRE()
                
                loss2 = nnloss(images,labels)
                
                loss = loss1 + loss2
                
                print(loss.item())

                net.zero_grad()
                loss.backward(retain_graph=True)
#                 print(BRE.lambda_prior_.grad)
#                 print(list(Z.grad for Z in list(net.parameters())))
                weights_grad = torch.cat(list(Z.grad.view(-1) for Z in list(net.parameters())), dim=0)
    
                BRE.flat_params.grad += weights_grad
                BRE.sigma_posterior_.grad += weights_grad * nnloss.noise 
                
                optimizer.step()
                optimizer.zero_grad()
                
#                 norm_change.append(torch.norm(BRE.flat_params - initial_weights,p=2))
#                 print(norm_change)
#     plt.figure(figsize=(16,10), dpi= 80)
#     plt.plot(range(501), norm_change, color='tab:red')
                
if __name__ == '__main__':
    torch.manual_seed(300)
    if torch.cuda.is_available():
        main(test_cuda=True)
    main(test_cuda=False)
    
