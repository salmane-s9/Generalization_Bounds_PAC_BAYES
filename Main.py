import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from PacBayes_Loss import PacBayesLoss
from NN_loss import mnnLoss
from utils import *
from Mnist_dataset import binary_mnist_loader
from Architectures import * 
import pickle
import os  
from functools import reduce


def main(model_name, test_cuda=False):

    print('-'*80)
    
    device = torch.device("cuda" if test_cuda else "cpu")
    INPUT_SIZE = 784
    NUM_CLASSES = 2
    if (model_name[1]=='-'): 
        NB_LAYERS, HIDDEN_SIZE=1, int(model_name[2:])
    else:
        NB_LAYERS, HIDDEN_SIZE = int(model_name[1]), int(model_name[3:])
    
    
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    NUM_EPOCHS = 20 if model_name[0]=='T' else 100
    BATCH_SIZE = 100

    # Define the model of a network which weights we will optimize
    initial_net = create_network(NB_LAYERS, INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
    
    weight_path = 'SGD_solutions/{}.ckpt'.format(model_name)
    
    if os.path.isfile(weight_path):
        net = load_train_weights(initial_net, weight_path)
    else:
        print('Network not already trained')
        print('\n Starting Training for network : '+model_name)
        train, test = binary_mnist_loader(batch_size=BATCH_SIZE, shuffle=False, random_labels=(model_name[0]=='R'))
        run(model_name, initial_net, train, test, LEARNING_RATE, MOMENTUM, NUM_EPOCHS, device)
        print('Traininig done for network : '+model_name)
        net = load_train_weights(initial_net, weight_path)

    
    train_loader, test_loader = binary_mnist_loader(batch_size=1, shuffle=False, random_labels=(model_name[0]=='R'))

    conf_param = 0.025 
    Precision = 100 
    bound = 0.1 
    data_size = 55000
    # n_mtcarlo_approx = 150000
    n_mtcarlo_approx = 10
    delta_prime = 0.01
    if (model_name[0]=='R'):
        learning_rate = 0.0001
    else:
        learning_rate = 0.001
    

    lambda_prior = torch.tensor(-3., device=device).requires_grad_()
    sigma_posterior = torch.abs(parameters_to_vector(net.parameters())).to(device).requires_grad_()

    flat_params = parameters_to_vector(net.parameters())
    BRE = PacBayesLoss(lambda_prior, sigma_posterior, net, flat_params, conf_param, Precision, bound, data_size, device).to(device)

    optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, BRE.parameters()), lr=learning_rate, alpha=0.9)
    criterion = nn.CrossEntropyLoss()
    nnloss = mnnLoss(criterion, BRE.flat_params, BRE.sigma_posterior_, net, BRE.d_size, device)

    if (model_name[0]=='R'):
        epochs = 8
    else:
        epochs = 4
    print("==> Starting PAC-Bayes bound optimization")

    mean_losses, BRE_loss, KL_value, NN_loss_final, norm_weights, norm_sigma, norm_lambda = (list() for i in range(7))
    for epoch in np.arange(1, epochs+1):   
        NN_loss = list()
        print(" \n Epoch {} :  ".format(epoch), end="\n")
        if ((epoch == 4) & (model_name[0]=='T')):
            print("==> Changing Learning rate from {} to {}".format(learning_rate, learning_rate/10))
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate/10
            
        for i, (images, labels) in enumerate(train_loader):
            print("\r Progress: {}%".format(100 * i // BRE.data_size), end="")

            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            loss1 = BRE()
            loss1.backward(retain_graph=True)

            loss2 = nnloss(images, labels)
            loss = loss1 + loss2
            NN_loss.append(loss2)

            if (((100 * i // BRE.data_size) - (100 * (i-1) // BRE.data_size)) != 0 and i != 0): 
                print('\t Mean loss : {} \r'.format(sum(mean_losses)/len(mean_losses)))
                mean_losses = []
            else:
                mean_losses.append(loss.item())
                
            net.zero_grad()
            loss2.backward()

            weights_grad = torch.cat(list(Z.grad.view(-1) for Z in list(nnloss.model.parameters())), dim=0)
            BRE.flat_params.grad += weights_grad
            BRE.sigma_posterior_.grad += weights_grad * nnloss.noise 

            optimizer.step()
            optimizer.zero_grad()

        BRE_loss.append(loss1.item())
        KL_value.append(BRE.kl_value)
        NN_loss_final.append(reduce(lambda a, b : a + b, NN_loss) / len(NN_loss))
        norm_weights.append(torch.norm(BRE.flat_params.clone().detach(), p=2))
        norm_sigma.append(torch.norm(BRE.sigma_posterior_.clone().detach(), p=2))
        norm_lambda.append(torch.abs(BRE.lambda_prior_.clone().detach()))
    
    print("\n==> Optimization done ")
    print("\n==> Saving Parameters... ")
    with open('./PAC_solutions/' + str(model_name) + '_BRE_flat_params.pickle', 'wb') as handle:
        pickle.dump(BRE.flat_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./PAC_solutions/' + str(model_name) + '_BRE_sigma_posterior.pickle', 'wb') as handle:
        pickle.dump(BRE.sigma_posterior_, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./PAC_solutions/' + str(model_name) + '_BRE_lambda_prior.pickle', 'wb') as handle:
        pickle.dump(BRE.lambda_prior_, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plot_results(model_name, BRE_loss, Kl_value, NN_loss_final, norm_weights, norm_sigma, norm_lambda)

    print("\n==> Calculating SNN train error and PAC Bayes bound :", end='\t')
    snn_train_error, Pac_bound = BRE.compute_bound(train_loader, delta_prime, n_mtcarlo_approx) 
    print("Done")
    print("\n==> Calculating SNN test error :", end='\t')
    snn_test_error = BRE.SNN_error(test_loader, delta_prime, n_mtcarlo_approx)
    print("Done")
    
    with open('./PAC_solutions/' + str(model_name) + '_FinalPac_bound.pickle', 'wb') as handle:
        pickle.dump(Pac_bound, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('\n Epoch {} Finished \t SNN_Train Error: {:.4f}\t SNN_Test Error: {:.4f} \t PAC-bayes Bound: {:.4f}\r'.format(epoch, snn_train_error,
                snn_test_error, Pac_bound))
    

if __name__ == '__main__':

    main(model_name='T-600', test_cuda=torch.cuda.is_available())
