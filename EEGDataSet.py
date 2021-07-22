from torch.utils.data import Dataset
from dataloader import read_bci_data
from torch.utils.data import DataLoader

import os
import torch



class EEGDataset(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        if dataset == "training":
            self.data, self.label, *_ = read_bci_data()
        elif dataset == "testing":
            *_, self.data, self.label = read_bci_data()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


if __name__ == "__main__":
    trainDataset = EEGDataset("training")
    testDataset = EEGDataset("testing")
    train_dataloader = DataLoader(trainDataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(testDataset, batch_size=64, shuffle=True)

    features, labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {features.size()}")
    print(f"Labels batch shape: {labels.size()}")
    print(labels)
