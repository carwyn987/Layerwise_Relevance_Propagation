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
        input_layer_out, activation_out, output_out = self.model.forward_save_intermediate(img.reshape(-1, 28*28).to(self.device))
        activation_out = np.squeeze(activation_out)
        print("Activation out: ", activation_out.shape)

        # For all nodes in hidden layer, compute relevance (ISSUE EXISTS IN THIS CODE)
        for i in range(hidden_size):
            total_rel = 0
            denominator = 0
            for j in range(output_size):
                denominator += activation_out[i]*self.model.state_dict()["output_layer.weight"][j][i] # + self.model.state_dict()["output_layer.bias"][j]
            for j in range(output_size):
                term_rel = activation_out[i]*self.model.state_dict()["output_layer.weight"][j][i] # + self.model.state_dict()["output_layer.bias"][j]

            total_rel = term_rel / denominator
            print("Total rel: ", total_rel)
            hidden_rel[i] = total_rel

        # Compute Input Relevences


        print("\nSum input: ", np.sum(input_rel), ". Sum hidden: ", np.sum(hidden_rel), ", Sum output: ", np.sum(output_rel), "\n")

        # Assert relevance conservation properties
        assert np.sum(input_rel) == np.sum(hidden_rel) and np.sum(hidden_rel) ==  np.sum(output_rel)

        return np.zeros((28,28))