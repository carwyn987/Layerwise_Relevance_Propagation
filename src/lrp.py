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

        

        return np.zeros((28,28))