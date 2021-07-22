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


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()

        self.firstConv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(1, 51),
                stride=(1, 1),
                padding=(0, 25),
                bias=False,
                device=device,
            ),
            nn.BatchNorm2d(
                num_features=16,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(2, 1),
                stride=(1, 1),
                groups=16,
                bias=False,
            ),
            nn.BatchNorm2d(
               num_features=32,
               eps=1e-05,
               momentum=0.1,
               affine=True,
               track_running_stats=True,
            ),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, 15),
                stride=(1, 1),
                padding=(0, 7),
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=32,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(
                kernel_size=(1, 8),
                stride=(1, 8),
                padding=0,
            ),
            nn.Dropout(p=0.25)
        )

        self.classify = nn.Sequential(
            nn.Linear(
                in_features=736,
                out_features=2,
                bias=True,
            )
        )

    def forward(self, x):
        firstFeature = self.firstConv(x)
        depthFeature = self.depthwiseConv(firstFeature)
        finalFeature = self.separableConv(depthFeature)
        prediction = self.classify(finalFeature)
        return prediction


if __name__ == "__main__":
    net = EEGNet()
    print(net)
    # model = TestNetwork().to(device)
    # print(model)
    #
    # X = torch.rand(1, 2, device=device)
    # pred = model(X)
    # print(f"model predict: {pred}")