import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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
    def __init__(self, activation):
        super(EEGNet, self).__init__()

        self.activationDict = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'ELU': nn.ELU(),
        }

        self.firstConv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(1, 51),
                stride=(1, 1),
                padding=(0, 20),
                bias=False,
                dtype=torch.float,
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
                kernel_size=(2, 5),
                stride=(1, 2),
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
            self.activationDict[activation],
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.5)
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
            self.activationDict[activation],
            nn.AvgPool2d(
                kernel_size=(1, 8),
                stride=(1, 8),
                padding=0,
            ),
            nn.Dropout(p=0.5)
        )

        self.flatten = nn.Flatten()
        self.classify = nn.Sequential(
            nn.Linear(
                in_features=352,
                out_features=2,
                bias=True,
            )
        )

    def forward(self, x):
        firstFeature = self.firstConv(x)
        depthFeature = self.depthwiseConv(firstFeature)
        finalFeature = self.separableConv(depthFeature)
        flattenFeature = self.flatten(finalFeature)
        prediction = self.classify(flattenFeature)
        return prediction


class EEGNetOrigin(nn.Module):
    def __init__(self, activation):
        super(EEGNetOrigin, self).__init__()

        self.activationDict = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'ELU': nn.ELU(),
        }

        self.firstConv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(1, 51),
                stride=(1, 1),
                padding=(0, 25),
                bias=False,
                dtype=torch.float,
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
            self.activationDict[activation],
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
            self.activationDict[activation],
            nn.AvgPool2d(
                kernel_size=(1, 8),
                stride=(1, 8),
                padding=0,
            ),
            nn.Dropout(p=0.25)
        )

        self.flatten = nn.Flatten()
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
        flattenFeature = self.flatten(finalFeature)
        prediction = self.classify(flattenFeature)
        return prediction


class DeepConvNet(nn.Module):
    def __init__(self, activation):
        super(DeepConvNet, self).__init__()

        self.activationDict = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'ELU': nn.ELU(),
        }

        self.doubleConv = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 10), padding="valid"),
            nn.BatchNorm2d(25),
            nn.Conv2d(25, 25, kernel_size=(2, 5), padding="valid"),
            nn.BatchNorm2d(25),
            self.activationDict[activation],
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
        )

        self.secondConv = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 10), padding="valid"),
            nn.BatchNorm2d(50),
            self.activationDict[activation],
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
        )

        self.thirdConv = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 10), stride=(1, 2), padding="valid"),
            nn.BatchNorm2d(100),
            self.activationDict[activation],
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
        )

        self.fourthConv = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 10), padding="valid"),
            nn.BatchNorm2d(200),
            self.activationDict[activation],
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
        )

        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            # nn.Linear(8600, 500),
            # nn.ReLU(),
            # nn.Linear(500, 2),
            nn.Linear(3200, 2),
            nn.Dropout(p=0.25)
        )

    def forward(self, x):
        firstFeature = self.doubleConv(x)
        feature = self.secondConv(firstFeature)
        feature = self.thirdConv(feature)
        feature = self.fourthConv(feature)
        flatten = self.flatten(feature)
        pred = self.dense(flatten)
        return pred


class DeepConvNetOrigin(nn.Module):
    def __init__(self, activation):
        super(DeepConvNetOrigin, self).__init__()

        self.activationDict = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'ELU': nn.ELU(),
        }

        self.doubleConv = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), padding="valid"),
            nn.Conv2d(25, 25, kernel_size=(2, 1), padding="valid"),
            nn.BatchNorm2d(25),
            self.activationDict[activation],
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
        )

        self.secondConv = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), padding="valid"),
            nn.BatchNorm2d(50),
            self.activationDict[activation],
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
        )

        self.thirdConv = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), padding="valid"),
            nn.BatchNorm2d(100),
            self.activationDict[activation],
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
        )

        self.fourthConv = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), padding="valid"),
            nn.BatchNorm2d(200),
            self.activationDict[activation],
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
        )

        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            # nn.Linear(8600, 500),
            # nn.ReLU(),
            # nn.Linear(500, 2),
            nn.Linear(8600, 2),
        )

    def forward(self, x):
        firstFeature = self.doubleConv(x)
        feature = self.secondConv(firstFeature)
        feature = self.thirdConv(feature)
        feature = self.fourthConv(feature)
        flatten = self.flatten(feature)
        pred = self.dense(flatten)
        return pred


def chooseModel(device):
    print("Choose the model you want to train:")
    print("1. EEGNet")
    print("2. DeepConvNet")
    netStruct = int(input())

    print("Choose the activation function:")
    print("1. ReLU")
    print("2. LeakyReLU")
    print("3. ELU")
    activationChoice = int(input())

    if activationChoice == 1:
        activation = "ReLU"
    elif activationChoice == 2:
        activation = "LeakyReLU"
    else:
        activation = "ELU"

    # Build CNN
    if netStruct == 1:
        # model = EEGNet(activation).to(device)
        model = EEGNetOrigin(activation).to(device)
    else:
        model = DeepConvNetOrigin(activation).to(device)

    return model


if __name__ == "__main__":
    net = DeepConvNet('ReLU')
    print(net)
