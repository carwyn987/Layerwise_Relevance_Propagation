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

    def get_lrp_image(self, img, lrp_rule, epsilon, gamma):
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

        # Compute Relevences

        # First get activations at hidden layer
        flattened_image = np.squeeze(img.reshape(-1, 28*28)).to(self.device)
        input_layer_out, activation_out, output_out = self.model.forward_save_intermediate(flattened_image)
        activation_out = np.squeeze(activation_out)
        input_layer_out = np.squeeze(input_layer_out)

        if lrp_rule == "0":
            epsilon = 0
            return self.lrp_0_and_epsilon(epsilon, input_size, hidden_size, output_size, activation_out, input_rel, hidden_rel, flattened_image)
        elif lrp_rule == "epsilon":
            return self.lrp_0_and_epsilon(epsilon, input_size, hidden_size, output_size, activation_out, input_rel, hidden_rel, flattened_image)
        elif lrp_rule == "gamma":
            epsilon = 0.1
            return self.lrp_gamma(epsilon, gamma, input_size, hidden_size, output_size, activation_out, input_rel, hidden_rel, flattened_image)
        elif lrp_rule == "composite":
            return self.lrp_composite(epsilon, gamma, input_size, hidden_size, output_size, activation_out, input_rel, hidden_rel, flattened_image)
        else:
            raise NotImplementedError("LRP Rule was not recognized.")
        

    def lrp_0_and_epsilon(self, epsilon, input_size, hidden_size, output_size, activation_out, input_rel, hidden_rel, flattened_image):
        # For all nodes in hidden layer, compute relevance
        for i in range(hidden_size):
            denominator = epsilon
            for j in range(output_size):
                denominator += activation_out[i]*self.model.state_dict()["output_layer.weight"][j][i] + self.model.state_dict()["output_layer.bias"][j]
            if epsilon == 0 and denominator <= 0.00001:
                denominator = 1
            term_rel = 0            
            for j in range(output_size):
                term_rel += (activation_out[i]*self.model.state_dict()["output_layer.weight"][j][i] + self.model.state_dict()["output_layer.bias"][j])/denominator

            hidden_rel[i] = term_rel

        # Compute Input Relevences
        # For all nodes in hidden layer, compute relevance
        for i in range(input_size):
            denominator = epsilon
            for j in range(hidden_size):
                denominator += flattened_image[i]*self.model.state_dict()["input_layer.weight"][j][i] + self.model.state_dict()["input_layer.bias"][j]
            if epsilon == 0 and denominator <= 0.00001:
                denominator = 1
            term_rel = 0
            for j in range(hidden_size):
                term_rel += (flattened_image[i]*self.model.state_dict()["input_layer.weight"][j][i] + self.model.state_dict()["input_layer.bias"][j])/denominator

            input_rel[i] = term_rel

        # print(input_rel)
        # print("\nSum input: ", np.sum(input_rel), ". Sum hidden: ", np.sum(hidden_rel), ", Sum output: ", np.sum(output_rel), "\n")
        # print("min: ", np.min(input_rel), ", max: ", np.max(input_rel))
        # input_rel = (255*(input_rel - np.min(input_rel))/np.ptp(input_rel)).astype(int)
        # print("After normalizing to (0,255), min: ", np.min(input_rel), ", max: ", np.max(input_rel))
        # Assert relevance conservation properties
        # assert np.sum(input_rel) == np.sum(hidden_rel) and np.sum(hidden_rel) ==  np.sum(output_rel)
        
        return input_rel.reshape((28,28))

    def lrp_gamma(self, epsilon, gamma, input_size, hidden_size, output_size, activation_out, input_rel, hidden_rel, flattened_image):
        # For all nodes in hidden layer, compute relevance
        for i in range(hidden_size):
            denominator = 0
            for j in range(output_size):
                denominator += activation_out[i]*self.model.state_dict()["output_layer.weight"][j][i] + self.model.state_dict()["output_layer.bias"][j]
                if self.model.state_dict()["input_layer.weight"][j][i] > 0:
                    denominator += gamma*self.model.state_dict()["input_layer.weight"][j][i]
            if denominator <= 0.00001:
                denominator = 1
            term_rel = 0            
            for j in range(output_size):
                term_rel += (activation_out[i]*self.model.state_dict()["output_layer.weight"][j][i] + self.model.state_dict()["output_layer.bias"][j])
                if self.model.state_dict()["input_layer.weight"][j][i] > 0:
                    term_rel += gamma*self.model.state_dict()["input_layer.weight"][j][i]
                term_rel /= denominator

            hidden_rel[i] = term_rel

        # Compute Input Relevences
        # For all nodes in hidden layer, compute relevance
        for i in range(input_size):
            denominator = 0
            for j in range(hidden_size):
                denominator += flattened_image[i]*self.model.state_dict()["input_layer.weight"][j][i] + self.model.state_dict()["input_layer.bias"][j]
                if self.model.state_dict()["input_layer.weight"][j][i] > 0:
                    denominator += gamma*self.model.state_dict()["input_layer.weight"][j][i]
            if denominator <= 0.00001:
                denominator = 1
            term_rel = 0
            for j in range(hidden_size):
                term_rel += (flattened_image[i]*self.model.state_dict()["input_layer.weight"][j][i] + self.model.state_dict()["input_layer.bias"][j])
                if self.model.state_dict()["input_layer.weight"][j][i] > 0:
                    term_rel += gamma*self.model.state_dict()["input_layer.weight"][j][i]
                term_rel /= denominator

            input_rel[i] = term_rel
        
        return input_rel.reshape((28,28))

    def lrp_composite(self, epsilon, gamma, input_size, hidden_size, output_size, activation_out, input_rel, hidden_rel, flattened_image):
        # For all nodes in hidden layer, compute relevance
        for i in range(hidden_size):
            denominator = epsilon
            for j in range(output_size):
                denominator += activation_out[i]*self.model.state_dict()["output_layer.weight"][j][i] + self.model.state_dict()["output_layer.bias"][j]
                if self.model.state_dict()["input_layer.weight"][j][i] > 0:
                    denominator += gamma*self.model.state_dict()["input_layer.weight"][j][i]
            if epsilon == 0 and denominator <= 0.00001:
                denominator = 1
            term_rel = 0            
            for j in range(output_size):
                term_rel += (activation_out[i]*self.model.state_dict()["output_layer.weight"][j][i] + self.model.state_dict()["output_layer.bias"][j])
                if self.model.state_dict()["input_layer.weight"][j][i] > 0:
                    term_rel += gamma*self.model.state_dict()["input_layer.weight"][j][i]
                term_rel /= denominator

            hidden_rel[i] = term_rel

        # Compute Input Relevences
        # For all nodes in hidden layer, compute relevance
        for i in range(input_size):
            denominator = epsilon
            for j in range(hidden_size):
                denominator += flattened_image[i]*self.model.state_dict()["input_layer.weight"][j][i] + self.model.state_dict()["input_layer.bias"][j]
            if epsilon == 0 and denominator <= 0.00001:
                denominator = 1
            term_rel = 0
            for j in range(hidden_size):
                term_rel += (flattened_image[i]*self.model.state_dict()["input_layer.weight"][j][i] + self.model.state_dict()["input_layer.bias"][j])/denominator

            input_rel[i] = term_rel

        
        return input_rel.reshape((28,28))