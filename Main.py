
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.autograd import Variable
from math import log , pi
import numpy as np

from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

from PacBayes_Loss import PacBayesLoss
from utils import *


# In[2]:


INPUT_SIZE = 784
HIDDEN_SIZE = [300 , 8 , 1200]
NUM_CLASSES = 2
NUM_EPOCHS = 20
# BATCH_SIZE = np.size(x_train)
LEARNING_RATE = 0.01
MOMENTUM = 0.9

Models = {'T-600' :FeedForwardNeuralNet(INPUT_SIZE, HIDDEN_SIZE[1], NUM_CLASSES),
          'T-1200':FeedForwardNeuralNet(INPUT_SIZE, HIDDEN_SIZE[2], NUM_CLASSES),
          'T2-300':FeedForwardNeuralNet2(INPUT_SIZE, HIDDEN_SIZE[0], NUM_CLASSES),
          'T2-600':FeedForwardNeuralNet2(INPUT_SIZE, HIDDEN_SIZE[1], NUM_CLASSES),
          'T2-1200':FeedForwardNeuralNet2(INPUT_SIZE, HIDDEN_SIZE[2], NUM_CLASSES),
          'T3-600' :FeedForwardNeuralNet3(INPUT_SIZE, HIDDEN_SIZE[1], NUM_CLASSES) }

net = Models['T-600']


# In[29]:


def alterning_targets(targets,label1_elements,label2_elements):
    '''
    We Change the classification task :
    We produce a binary classification problem by mapping :
    numbers {0,1,2,3,4} to label 0 and {5,6,7,8,9} to label 1
    '''
    new_targets = targets.copy()
    new_targets[np.isin(new_targets, label1_elements)] = 0
    new_targets[np.isin(new_targets, label2_elements)] = 1
    
    return new_targets

import tensorflow as tf
# Importing Tensorflow Dataset MNIST :
label1_elements = np.arange(0,5)
label2_elements = np.arange(5,10)
mnist = tf.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist
x_train , y_train = x_train[:55000] , alterning_targets(y_train,label1_elements,label2_elements)[:55000]
x_test , y_test = x_test , alterning_targets(y_test,label1_elements,label2_elements)

class CustomMNIST(Dataset):
    def __init__(self, data ,targets, height, width, transform=None):
        """
        Constructing a custom Dataset of MNIST on Pytorch
        Args:
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = data
        self.labels = targets
        self.height = height
        self.width = width
        self.transform = transform

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        # Read each 784 pixels and reshape the 1D array ([784]) to 2D array ([28,28])
        img_as_np = self.data[index].reshape(self.height, self.height).astype('uint8')
        # Convert image from numpy array to PIL image, mode 'L' is for grayscale
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
        # Transform image to tensor
        if self.transform is not None:
            img_as_tensor = self.transform(img_as_img)
        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data)
    
    
transformations = transforms.Compose([transforms.ToTensor()])
custom_mnist_train =     CustomMNIST(x_train,y_train,
                             28, 28,
                             transformations)
train_loader = torch.utils.data.DataLoader(dataset=custom_mnist_train,
                                                    batch_size=1,
                                                    shuffle=True)

# testing data and loader
custom_mnist_test =     CustomMNIST(x_test,y_test,
                             28, 28,
                             transformations)
test_loader = torch.utils.data.DataLoader(dataset=custom_mnist_test,
                                                    batch_size=1,
                                                    shuffle=False)


# In[30]:


def main(test_cuda=False):
    print('-'*80)
    device = torch.device("cuda" if test_cuda else "cpu")
    net = Models['T-600']
    conf_param=0.025 
    Precision= 100 
    bound=0.1 
    data_size= 55000
    
    lambda_prior_ = torch.tensor(-3. ,device=device).requires_grad_()
    
    sigma_posterior_ = torch.abs(parameters_to_vector(net.parameters())).requires_grad_()

    ct = PacBayesLoss(lambda_prior_, sigma_posterior_, net, conf_param, Precision, bound, 
                      data_size).to(device)
    
    optimizer = torch.optim.RMSprop(ct.parameters(), lr = 0.001)
    criterion  = nn.CrossEntropyLoss()
    
    
    for i, (images, labels) in enumerate(train_loader):
            if i > 1:
                break
            print("\r{}%".format(100 * i // ct.data_size), end="")
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            
            loss1 = ct(net.parameters())
            
            print(loss1.item())
#             loss1.backward(retain_graph=True)
            
            print(ct.flat_params.grad)
            
            modified_parameters = ct.flat_params + torch.randn(ct.d_size) * torch.exp(2 * ct.sigma_posterior_).abs()
            indi = 0
            for name,ind,shape_ in network_params(net):
                net.state_dict()[name].data.copy_(modified_parameters[indi:indi+ind].view(shape_)) 
                indi = ind
                
        
            
            outputs = net(images)
            loss2 = criterion(outputs.float(), labels.long())
                
            loss = loss1 + loss2
            
            loss2.backward()
            print(ct.flat_params.grad)
            
            
#             optimizer.step()

if __name__ == '__main__':
    torch.manual_seed(500)
    main(test_cuda=False)
    if torch.cuda.is_available():
        main(test_cuda=True)
