import torch

def accuracy(test_loader, model, device):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item() 

        acc = 100.0 * n_correct / n_samples
    
    return (f'Accuracy of the network on the 10000 test images: {acc} %')