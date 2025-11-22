import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



train_set = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

print (type(train_set))
print (type(train_set[0]))
print (len(train_set))
print (type(train_set[0][0]))
print (train_set[0][0].shape)
print (type(train_set[1]))

test_set = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

print (type(test_set))
print (type(test_set[0]))
print (len(test_set))
print (type(test_set[0][0]))
print (test_set[0][0].shape)
print (type(test_set[1]))


train_loader = DataLoader(train_set, batch_size=32, shuffle=False)

for image, label in train_loader:
    flattened_image = image.reshape(image.size(0),-1)
    print (flattened_image.shape, label.shape)

#print (len(loader))

#print (loader[1])