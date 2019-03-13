import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from PacBayes_Loss import PacBayesLoss
from NN_loss import mnnLoss
from utils import *
from Mnist_dataset import binary_mnist_loader
from Architectures import * 


def main(test_cuda=False, weight_path=None):

    print('-'*80)
    device = torch.device("cuda" if test_cuda else "cpu")
    INPUT_SIZE = 784
    HIDDEN_SIZE = [300, 600, 1200]
    NUM_CLASSES = 2

    # Define the model of a network which weight we will optimize
    initial_net = FeedForwardNeuralNet(INPUT_SIZE, HIDDEN_SIZE[1], NUM_CLASSES)

    if weight_path is not None:
        net = load_train_weights(initial_net, weight_path)
    else:
        net = initial_net

    train_loader, test_loader = binary_mnist_loader()

    conf_param = 0.025 
    Precision = 100 
    bound = 0.1 
    data_size = 55000
    # n_mtcarlo_approx = 150000
    n_mtcarlo_approx = 2
    delta_prime = 0.01
    learning_rate = 0.001

    lambda_prior = torch.tensor(-3., device=device).requires_grad_()
    sigma_posterior = torch.abs(parameters_to_vector(net.parameters())).requires_grad_()

    BRE = PacBayesLoss(lambda_prior, sigma_posterior, net, conf_param, Precision, bound, data_size).to(device)

    optimizer = torch.optim.RMSprop(BRE.parameters(), lr=learning_rate, alpha=0.9)
    criterion = nn.CrossEntropyLoss()
    nnloss = mnnLoss(criterion, BRE.flat_params, BRE.sigma_posterior_, net, BRE.d_size)
    epochs = 4

    mean_losses = []
    for epoch in np.arange(1, epochs+1):   
        print(" \n Epoch {} :  ".format(epoch), end="\n")
        if (epoch == 4): 
            print("==> Changing Learning rate from {} to {}".format(learning_rate, learning_rate/10))
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate/10
            
        for i, (images, labels) in enumerate(train_loader):
            
            print("\r Progress: {}%".format(100 * i // BRE.data_size), end="")

            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            loss = BRE() + nnloss(images, labels)

            if (((100 * i // BRE.data_size) - (100 * (i-1) // BRE.data_size)) != 0 and i!=0): 
                print('\t Mean loss : {} \r'.format(sum(mean_losses)/len(mean_losses)))
                mean_losses = []
            else:
                mean_losses.append(loss.item())
                
            net.zero_grad()
            loss.backward(retain_graph=True)

            weights_grad = torch.cat(list(Z.grad.view(-1) for Z in list(net.parameters())), dim=0)
            BRE.flat_params.grad += weights_grad
            BRE.sigma_posterior_.grad += weights_grad * nnloss.noise 

            optimizer.step()
            optimizer.zero_grad()

        snn_train_error, Pac_bound = BRE.compute_bound(train_loader, delta_prime, n_mtcarlo_approx, device)     
        snn_test_error = BRE.SNN_error(test_loader, delta_prime, n_mtcarlo_approx, device)

        print('\n Epoch {} Finished \t SNN_Train Error: {:.4f}\t SNN_Test Error: {:.4f} \t PAC-bayes Bound: {:.4f}\r'.format(epoch, snn_train_error,
                snn_test_error, Pac_bound))
    
if __name__ == '__main__':
    # torch.manual_seed(300)
    weight_path = 'SGD_solutions/T-600.ckpt'
    if torch.cuda.is_available():
        main(test_cuda=True)
    else:
        main(test_cuda=False)