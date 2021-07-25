from model import *
from deviceCheck import device
from EEGDataSet import EEGDataset
from loop import epoch_loop


if __name__ == "__main__":
    structList = ["EEG", "DCN"]
    activationList = ["ReLU", "LeakyReLU", "ELU"]

    trainData = EEGDataset("training")
    testData = EEGDataset("testing")
    trainLoader = DataLoader(trainData, batch_size=64, shuffle=True)
    testLoader = DataLoader(testData, batch_size=64, shuffle=True)

    for netStruct in structList:
        for activation in activationList:
            print(f"{netStruct} with {activation} training start")
            if netStruct == "EEG":
                model = EEGNet(activation).to(device)
            else:
                model = DeepConvNet(activation).to(device)

            # Hyper parameter
            batch_size = 64
            learning_rate = 1e-3
            epochs = 400
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            loss_function = nn.CrossEntropyLoss()

            epoch_loop(epochs, trainLoader, testLoader, model, loss_function, optimizer)
            print('finish\n')