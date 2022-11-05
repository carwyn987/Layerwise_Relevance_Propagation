import torch
import torch.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, activation) -> None:
        super(SimpleNetwork, self).__init__()

        match activation:
            case "sigmoid":
                self.activation = nn.Sigmoid()
            case "tanh":
                self.activation = nn.Tanh()
            case "relu":
                self.activation = nn.ReLU()
            case _:
                self.activation = nn.Sigmoid()
        
        self.input_size = input_size
        self.output_size = 10
        self.input_layer = nn.Linear(input_size, hidden_size) 
        self.output_layer = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

    def get_loss_function(self, loss_function):
        match loss_function:
            case "cross-entropy":
                loss_f = nn.CrossEntropyLoss()
            case "mse":
                raise NotImplementedError
                loss_f = nn.MSELoss()
            case _:
                raise NotImplementedError("The loss function was not recognized")
        return loss_f
 
    def get_optimizer(self, optim, learning_rate):
        match optim:
            case "adam":
                optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate) 
            case "sgd":
                optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
            case _:
                raise NotImplementedError("The optimizer was not recognized")
        return optimizer