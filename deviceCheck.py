import torch

# Select computing device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))