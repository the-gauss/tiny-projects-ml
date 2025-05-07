from oxford import preprocessing
import json
import torch
import torchvision


class BaseLine(torch.nn.Module):
    def __init__(self, n_layers = 2 ,activation = torch.nn.ReLU()):
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=1)
        print(self.flatten)
        self.activation = activation
        self.n_layers = n_layers
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(3*224*224, 512),
            self.activation,
            *[torch.nn.Linear(512, 512), self.activation]*self.n_layers,
            torch.nn.Linear(512,100)
        )
        print(self.linear_relu_stack)

    def forward(self, x):
        x = self.flatten(x) # Flattens the second dimension (dim=1), leaves dim=0 intact
        logits = self.linear_relu_stack(x)  # Returns unnormalized [-infty, infty] outputs of the final layer
        return logits

