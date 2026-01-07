import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset






from utilities import MLP, train_model, evaluate, run_experiment



train_set = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)




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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



run_experiment(train_loader,
                   test_loader,
                   permutations,
                   #plt
                   )

run_experiment(train_loader,
                   test_loader,
                   permutations,
                   #plt,
                   superposition=True)

