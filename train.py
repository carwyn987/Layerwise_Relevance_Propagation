from tqdm import tqdm

def train(train_data_loader, model, device, criterion, optimizer, epochs):

    length_data_loader = len(train_data_loader)
    for epoch in range(epochs):
        average_loss = 0
        print("Epoch ", epoch)
        for i, (images, labels) in enumerate(tqdm(train_data_loader)):  
            # origin shape: [100, 1, 28, 28]
            # resized: [100, 784]
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            average_loss += loss.item()
            optimizer.step()
        print("Average loss: ", average_loss/length_data_loader, "\n")
