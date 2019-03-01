import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import tensorflow.keras.datasets.mnist as mnist_

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

def alterning_targets(targets, label1_elements, label2_elements):
    '''
    We Change the classification task :
    We produce a binary classification problem by mapping :
    numbers {0,1,2,3,4} to label 0 and {5,6,7,8,9} to label 1
    '''
    new_targets = targets.copy()
    new_targets[np.isin(new_targets, label1_elements)] = 0
    new_targets[np.isin(new_targets, label2_elements)] = 1
    
    return new_targets
def binary_mnist_loader():
    
    # Importing Tensorflow Dataset MNIST and binarizing targets:
    label1_elements = np.arange(0,5)
    label2_elements = np.arange(5,10)
    mnist = mnist_.load_data()
    (x_train, y_train), (x_test, y_test) = mnist
    x_train , y_train = x_train[:55000] , alterning_targets(y_train,label1_elements,label2_elements)[:55000]
    x_test , y_test = x_test , alterning_targets(y_test,label1_elements,label2_elements)

    transformations = transforms.Compose([transforms.ToTensor()])
    custom_mnist_train =         CustomMNIST(x_train,y_train,
                                 28, 28,
                                 transformations)
    train_loader = torch.utils.data.DataLoader(dataset=custom_mnist_train,
                                                        batch_size=1,
                                                        shuffle=True)

    # testing data and loader
    custom_mnist_test =         CustomMNIST(x_test,y_test,
                                 28, 28,
                                 transformations)
    test_loader = torch.utils.data.DataLoader(dataset=custom_mnist_test,
                                                        batch_size=1,
                                                        shuffle=False)
    
    return train_loader, test_loader
