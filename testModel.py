from loop import test_loop
from EEGDataSet import EEGDataset
from model import *
from networkLoader import load_network
from deviceCheck import device


if __name__ == "__main__":
    model = chooseModel(device)

    filename = input("Please input the network filename")
    model = load_network(model, filename)

    data = EEGDataset("testing")
    testLoader = DataLoader(data, batch_size=64)
    loss_fn = torch.nn.CrossEntropyLoss()

    test_loop(testLoader, model, loss_fn)
