import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self, input_dim = 28*28, hidden1=256, hidden2=256, num_classes = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc_out = nn.Linear(hidden2, num_classes)

    def forward(self, inputs, targets=None):

        flattened_inputs = inputs.reshape(inputs.size(0),-1)
        logits = F.relu(self.fc1(flattened_inputs))
        logits = F.relu(self.fc2(logits))
        logits = self.fc_out(logits)

        return logits

        


train_set = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)



train_loader = DataLoader(train_set, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlp = MLP().to(device)


for image, label in train_loader:
    batch_logits = mlp.forward(image)
    predicted_classes = torch.argmax(batch_logits, dim=1)
    print (predicted_classes)
   
   
    

#print (len(loader))

#print (loader[1])


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