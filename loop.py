import torch
from torch import nn
from torch.utils.data import DataLoader
from deviceCheck import device


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    corrects = 0

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Forward pass, model predict
        pred = model(X.float())
        loss = loss_fn(pred, y.long())

        # Count the among of correct prediction
        corrects += (pred.argmax(dim=1) == y).type(torch.long).sum().item()

        # Backward pass, parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    corrects = corrects / size
    return corrects * 100


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()    # Change to evaluation mode
    test_loss, corrects = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            # Forward pass, model predict
            pred = model(X.float())
            test_loss += loss_fn(pred, y.long()).item()
            # corrects += 1 if (pred.argmax(dim=1) == y) else 0
            corrects += (pred.argmax(dim=1) == y).type(torch.long).sum().item()

    test_loss /= num_batches
    corrects /= size
    print(f"Test Error: \n Accuracy: {(100*corrects):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return corrects * 100


def epoch_loop(epochs, train_loader, test_loader, model, loss_function, optimizer):
    trainAcs = [0] * (epochs + 1)
    testAcs = [0] * (epochs + 1)

    for e in range(1, epochs + 1):
        print(f"Epoch [{e}] start:")
        print("-------------------------------")
        trainAcs[e] = train_loop(train_loader, model, loss_function, optimizer)
        testAcs[e] = test_loop(test_loader, model, loss_function)
        print("-------------------------------\n")

    print(trainAcs)
    print(testAcs)
