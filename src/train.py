import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def train(train_data_loader, model, device, criterion, optimizer, epochs, path):

    # Define array for saving training data
    length_data_loader = len(train_data_loader)
    losses = np.zeros((length_data_loader*epochs))
    
    for epoch in range(epochs):
        average_loss = 0
        print("Epoch ", epoch)
        for i, (images, labels) in enumerate(tqdm(train_data_loader)):  
            # origin shape: [100, 1, 28, 28]
            # resized: [100, 784]
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            # Predict
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Update
            optimizer.zero_grad()
            loss.backward()
            average_loss += loss.item()
            optimizer.step()
            # Save
            losses[epoch*length_data_loader + i] = loss.item()

        print("Average loss: ", average_loss/length_data_loader, "\n")

    if path:
        # Create training loss figure
        plt.figure(1)
        plt.xlabel('Training iteration')
        plt.ylabel('Loss')
        plt.title('Loss over Training Iterations')
        plt.text(120, 1.1, str(length_data_loader) + ' iterations per epoch')
        plt.grid(True)
        plt.plot(np.arange(0,len(losses),1), losses)
        plt.savefig(path + "train_loss.png")