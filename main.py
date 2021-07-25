from model import *
from EEGDataSet import EEGDataset
from torch.utils.data import DataLoader
from torch import nn
from loop import *
from networkLoader import save_network
import torch

# Select computing device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


# Neuron network
# model = EEGNet().to(device)
model = DeepConvNet().to(device)
model = model.float()

# Hyper parameter
batch_size = 64
learning_rate = 1e-3
epochs = 400
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()


if __name__ == "__main__":
    trainDataset = EEGDataset("training")
    testDataset = EEGDataset("testing")

    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=True)

    for e in range(1, epochs + 1):
        print(f"Epoch [{e}] start:")
        print("-------------------------------")
        train_loop(trainLoader, model, loss_function, optimizer)
        test_loop(testLoader, model, loss_function)
        print("-------------------------------\n")

    print("Finish")

    print("Do you want to save the network?")
    print("1. yes")
    print("2. no")
    choice = input()

    if choice == 1:
        filename = input("Please input the filename")
        save_network(model, filename)
