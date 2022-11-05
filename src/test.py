import torch
import numpy as np
import matplotlib.pyplot as plt

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

def predictSample(path, test_loader, model, device):
    # Create training loss figure
    fig, axs = plt.subplots(10, 10, figsize=(8, 8))
    fig.suptitle('Test images sample predictions ( with highlighted errors )')
    
    with torch.no_grad():
        for classPredict in range(10):
            numchosen = 1
            for images, correct_labels in test_loader:
                model_imgs = images.reshape(-1, 28*28).to(device)
                labels = np.argmax(model(model_imgs), axis=1)
                for sample_indx in range(len(labels)):
                    if numchosen > 10:
                        break
                    if labels[sample_indx] == classPredict:
                        img = np.squeeze(images[sample_indx])
                        # If predicted incorrect label, highlight it
                        if not labels[sample_indx] == correct_labels[sample_indx]:
                            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
                            img[:,:,1:2] = 0
                        axs[classPredict, numchosen-1].set_axis_off()
                        axs[classPredict, numchosen-1].imshow(img)
                        numchosen += 1
                if numchosen > 10:
                        break

    fig.savefig(path + "predict_sample.png")