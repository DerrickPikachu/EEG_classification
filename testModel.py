from loop import test_loop
from EEGDataSet import EEGDataset
from model import *
from networkLoader import load_network


if __name__ == "__main__":
    print("Choose network structure:")
    print("1. EEGNet")
    print("2. DeepConvNet")
    netStructure = input()

    print("Choose the activation function:")
    print("1. ReLU")
    print("2. LeakyReLU")
    print("3. ELU")
    activationChoice = input()

    if activationChoice == 1:
        activation = "ReLU"
    elif activationChoice == 2:
        activation = "LeakyReLU"
    else:
        activation = "ELU"

    if netStructure == 1:
        model = EEGNet(activation)
    else:
        model = DeepConvNet(activation)

    filename = input("Please input the network filename")
    model = load_network(model, filename)

    data = EEGDataset("testing")
    testLoader = DataLoader(data, batch_size=64)
    loss_fn = torch.nn.CrossEntropyLoss()

    test_loop(testLoader, model, loss_fn)
