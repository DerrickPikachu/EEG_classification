from model import *
from deviceCheck import device
from EEGDataSet import EEGDataset
from loop import epoch_loop

import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("Which net you want to train?")
    print("1. EEGNet")
    print("2. DeepConvNet")
    netChoice = int(input())
    network = "EEGNet" if netChoice == 1 else "DeepConvNet"

    activationList = ["ReLU", "LeakyReLU", "ELU"]
    accuracyDict = {}

    # Hyper parameter
    batch_size = 16
    learning_rate = 1e-3
    epochs = 400
    loss_function = nn.CrossEntropyLoss()
    epochList = list(range(1, epochs + 1))

    trainData = EEGDataset("training")
    testData = EEGDataset("testing")
    trainLoader = DataLoader(trainData, batch_size=64, shuffle=True)
    testLoader = DataLoader(testData, batch_size=64, shuffle=True)

    for activation in activationList:
        print(f"{network} with {activation} training start")
        if network == "EEGNet":
            model = EEGNetOrigin(activation).to(device)
        else:
            model = DeepConvNetOrigin(activation).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        *accuracy, = epoch_loop(epochs, trainLoader, testLoader, model, loss_function, optimizer)
        print('finish\n')

        plt.plot(epochList, accuracy[0], label=activation + '_train', linewidth=1)
        plt.plot(epochList, accuracy[1], label=activation + '_test', linewidth=1)

    plt.legend()
    plt.title(f'Activation Function Comparison({network})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')

    plt.savefig(f"{network}.png", bbox_inches='tight')

