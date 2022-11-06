import torch

# Wrapper class for a model which contains lrp functions

# def decorator(cls, model_path, device):

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
            

    # return LRP