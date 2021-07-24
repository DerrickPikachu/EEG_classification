import torch
import torchvision.models as models
from torch.utils.data import DataLoader

from EEGDataSet import EEGDataset
from model import EEGNet
from loop import test_loop


def save_network(model, filename):
    filename += '.pth'
    torch.save(model.state_dict(), filename)


def load_network(model, filename):
    filename += '.pth'
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model


if __name__ == "__main__":
    net = EEGNet('ReLU')
    testData = EEGDataset("testing")
    testLoader = DataLoader(testData, batch_size=64)
    loss_fn = torch.nn.CrossEntropyLoss()

    net.eval()
    test_loop(testLoader, net, loss_fn)

    save_network(net, "test")

    net2 = EEGNet('ReLU')
    net2 = load_network(net2, "test")

    net2.eval()
    test_loop(testLoader, net2, loss_fn)
