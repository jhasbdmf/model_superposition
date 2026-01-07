import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset



import numpy as np
import matplotlib.pyplot as plt

from utilities import MLP, train_model, run_experiment





train_set = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)




#print (apply_pca_to_batch(pca_instances_loader))

test_set = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)




batch_size = 128
n_tasks = 10
input_dim = 784


permutations = torch.stack([torch.randperm(input_dim) for _ in range(n_tasks)])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)



test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
"""



# CIFAR-10: 50k train / 10k test, 10 classes (airplane, car, ...), 32x32 RGB

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 mean/std
])

train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

print(f"Train batches: {len(train_loader)}, Classes: {train_set.classes}")



batch_size = 128
n_tasks = 10
input_dim = 1024


permutations = torch.stack([torch.randperm(input_dim) for _ in range(n_tasks)])
"""


run_experiment(train_loader,
                   test_loader,
                   permutations,
                   input_dim
                   )

run_experiment(train_loader,
                   test_loader,
                   permutations,
                   input_dim=input_dim,
                   superposition=True)
