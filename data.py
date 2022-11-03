import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms

def load_data(batch_size, file):
    if file:
        raise NotImplementedError("MNIST file loading not implemented.")
    
    # Import MNIST dataset 
    train_dataset = torchvision.datasets.MNIST(root='./data', 
                                                train=True, 
                                                transform=transforms.ToTensor(),  
                                                download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', 
                                                train=False, 
                                                transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=False) 


    return train_loader, test_loader