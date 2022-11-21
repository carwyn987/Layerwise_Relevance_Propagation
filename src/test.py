import torch
import random
import numpy as np
import matplotlib.pyplot as plt

def standard_statistics(test_loader, model, device):
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
                labels = np.argmax(model(model_imgs).cpu(), axis=1)
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

def predictFeatureExtractionSample(path, img, lrp_img, label, model, device):
    with torch.no_grad():
        model_img = img.reshape(28*28).to(device)
        predicted_labels = model(model_img).cpu().numpy()
        predicted_label = np.argmax(predicted_labels)

        # Original image file paths
        img_save_file = path + "original" + ".png"
        txt_save_file = path + "original" + ".txt"
        np.savetxt(txt_save_file, predicted_labels)

        # Save original image
        fig, axs = plt.subplots(1, 1, figsize=(6, 3))
        fig.suptitle('Input image')
        axs.imshow(img)
        fig.savefig(img_save_file)
        
        # Add correct label
        with open(txt_save_file, "a") as myfile:
            myfile.write("Label: " + str(predicted_label))

        # Compute important image

        # Using the lrp-image, find area of maximum importance (values) and replace with mean of entire image
        mean_image = np.mean(model_img.cpu().numpy())
        
        # Replace top twentieth of the maximum values with mean
        lrp_img_save = np.array(lrp_img, copy=True) 
        important_feature_removal_img = model_img.cpu().numpy()
        for _ in range(150):
            important_feature_removal_img[np.argmin(lrp_img)] = mean_image
            lrp_img[np.argmin(lrp_img)] = lrp_img[np.argmax(lrp_img)]

        # Important image
        img_save_file = path + "important_feature_removal" + ".png"
        txt_save_file = path + "important_feature_removal" + ".txt"
        predicted_important_labels = model(torch.from_numpy(important_feature_removal_img).to(device)).cpu().numpy()
        np.savetxt(txt_save_file, predicted_important_labels)

        important_img = important_feature_removal_img.reshape((28, 28))

        # Save important image
        fig, axs = plt.subplots(1, 1, figsize=(6, 3))
        fig.suptitle('Important Feature Removal Image')
        axs.imshow(important_img)
        fig.savefig(img_save_file)

        # Compute important image
        
        # Replace top twentieth of the maximum values with mean
        unimportant_feature_removal_img = model_img.cpu().numpy()
        for _ in range(150):
            unimportant_feature_removal_img[np.argmax(lrp_img_save)] = mean_image
            lrp_img_save[np.argmax(lrp_img_save)] = lrp_img_save[np.argmin(lrp_img_save)]

        # Important image
        img_save_file = path + "unimportant_feature_removal" + ".png"
        txt_save_file = path + "unimportant_feature_removal" + ".txt"
        predicted_unimportant_labels = model(torch.from_numpy(unimportant_feature_removal_img).to(device)).cpu().numpy()
        np.savetxt(txt_save_file, predicted_unimportant_labels)

        unimportant_img = unimportant_feature_removal_img.reshape((28, 28))

        # Save important image
        fig, axs = plt.subplots(1, 1, figsize=(6, 3))
        fig.suptitle('Unimportant Feature Removal Image')
        axs.imshow(unimportant_img)
        fig.savefig(img_save_file)