import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn



class MLP(nn.Module):
    def __init__(self, superposition = False, n_tasks = 5, input_dim = 28*28, hidden1=256, hidden2=256, num_classes = 10):


        super().__init__()
        self.superposition = superposition
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc_out = nn.Linear(hidden2, num_classes)

        if superposition:
            self.context1 = torch.randint(0, 2, (n_tasks, hidden1)) * 2 - 1
            self.context2 = torch.randint(0, 2, (n_tasks, hidden2)) * 2 - 1

    def forward(self, inputs, task_id=None, targets=None):

        flattened_inputs = inputs.reshape(inputs.size(0),-1)
        logits = F.relu(self.fc1(flattened_inputs))

        #print (logits.shape)
        if self.superposition:
            logits = logits * self.context1[task_id]

        logits = F.relu(self.fc2(logits))

        if self.superposition:
            logits = logits * self.context2[task_id]


        logits = self.fc_out(logits)

        return logits
    

def train_model (model, train_loader, batch_size, n_epochs=2, n_tasks = 5):

    permutations = torch.stack([torch.randperm(model.input_dim) for _ in range(n_tasks)])
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()


    n_epochs = 2



    criterion = nn.CrossEntropyLoss()

    train_loss_history = []

    for t in range(n_tasks):

        for epoch in range(n_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(device)   # (B, 1, 28, 28)
                labels = labels.to(device)   # (B,)

                # Flatten and permute pixels
                B = images.size(0)
                images = images.view(B, -1)         # (B, 784)
                images = images[:, permutations[t]]            # (B, 784) permuted

                # Forward
                logits = model(images)
                loss = criterion(logits, labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * B
                #print (f"Running loss = {running_loss}")
                _, preds = logits.max(1)
                correct += preds.eq(labels).sum().item()
                total += B

            avg_loss = running_loss / total
            train_loss_history.append (avg_loss)
            acc = correct / total * 100.0
            #print(f"Task {task_id} | Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
            print(f"Task {t+1} | Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
        
    model.eval()

    return model, train_loss_history








        


train_set = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

batch_size = 32

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlp = MLP().to(device)


mlp, train_loss_history = train_model(model=mlp, train_loader=train_loader, batch_size=batch_size)

print (train_loss_history)



#for epochs in range(n_epochs):

#    for image, label in train_loader:
#        batch_logits = mlp.forward(image)
#        predicted_classes = torch.argmax(batch_logits, dim=1)
#        print (predicted_classes)
   

   
    

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