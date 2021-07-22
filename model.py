import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


class TestNetwork(nn.Module):
    def __init__(self):
        super(TestNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        prob = self.linear_relu_stack(x)
        return prob


if __name__ == "__main__":
    model = TestNetwork().to(device)
    print(model)

    X = torch.rand(1, 2, device=device)
    pred = model(X)
    print(f"model predict: {pred}")