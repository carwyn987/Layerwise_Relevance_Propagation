import numpy as np
import torch

# Wrapper class for a model which contains lrp functions

class LRP:

    def __init__(self, cls, model_path, device) -> None:
        self.model = cls.to(device)
        self.device = device
        print("Loading saved model.")
        self.model.load_state_dict(torch.load(model_path + "model.pt"))

    def single_pass(self, input):
        single_img = input.reshape(-1, 28*28).to(self.device)
        output = self.model(single_img).cpu()
        return output

    def get_lrp_image(self, img, lrp_rule):
        # Get original image
        outputs = self.single_pass(img).detach().numpy()
        single_output = np.argmax(outputs)
        print("Output: ", outputs, ", class argmax: ", single_output)

        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())


        input_size = self.model.state_dict()["input_layer.weight"].shape[1]
        hidden_size = self.model.state_dict()["input_layer.weight"].shape[0]
        output_size = self.model.state_dict()["output_layer.weight"].shape[0]

        print("Layers: \n", "  input size: ", input_size, "\n   hidden size: ", hidden_size, "\n   output size: ", output_size)

        # Define arrays to store intermediate layer relevances
        input_rel = np.zeros((input_size))
        hidden_rel = np.zeros((hidden_size))
        output_rel = outputs

        # Compute Hidden Relevences

        # First get activations at hidden layer
        flattened_image_device = np.squeeze(img.reshape(-1, 28*28)).to(self.device)
        input_layer_out, activation_out, output_out = self.model.forward_save_intermediate(flattened_image_device)
        activation_out = np.squeeze(activation_out)
        input_layer_out = np.squeeze(input_layer_out)
        print("Activation out: ", activation_out.shape, ", Input out: ", input_layer_out.shape)

        # For all nodes in hidden layer, compute relevance (ISSUE EXISTS IN THIS CODE)
        for i in range(hidden_size):
            total_rel = 0
            denominator = 0
            for j in range(output_size):
                denominator += activation_out[i]*self.model.state_dict()["output_layer.weight"][j][i] # + self.model.state_dict()["output_layer.bias"][j]
            for j in range(output_size):
                term_rel = activation_out[i]*self.model.state_dict()["output_layer.weight"][j][i] # + self.model.state_dict()["output_layer.bias"][j]

            total_rel = term_rel / denominator
            hidden_rel[i] = total_rel

        # Compute Input Relevences
        # For all nodes in hidden layer, compute relevance (ISSUE EXISTS IN THIS CODE)
        for i in range(input_size):
            total_rel = 0
            denominator = 0
            for j in range(hidden_size):
                denominator += flattened_image_device[i]*self.model.state_dict()["input_layer.weight"][j][i] # + self.model.state_dict()["output_layer.bias"][j]
            for j in range(hidden_size):
                term_rel = flattened_image_device[i]*self.model.state_dict()["input_layer.weight"][j][i] # + self.model.state_dict()["output_layer.bias"][j]

            
            total_rel = term_rel / denominator if denominator != 0 else 0
            input_rel[i] = total_rel

        print("\nSum input: ", np.sum(input_rel), ". Sum hidden: ", np.sum(hidden_rel), ", Sum output: ", np.sum(output_rel), "\n")
        print("min: ", np.min(input_rel), ", max: ", np.max(input_rel))
        input_rel = (255*(input_rel - np.min(input_rel))/np.ptp(input_rel)).astype(int)
        print("After normalizing to (0,255), min: ", np.min(input_rel), ", max: ", np.max(input_rel))
        # Assert relevance conservation properties
        # assert np.sum(input_rel) == np.sum(hidden_rel) and np.sum(hidden_rel) ==  np.sum(output_rel)

        return input_rel.reshape((28,28))