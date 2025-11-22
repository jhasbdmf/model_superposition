import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np



class MLP(nn.Module):
    def __init__(self, superposition = False, n_tasks = 5, input_dim = 28*28, hidden1=128, hidden2=128, num_classes = 10):


        super().__init__()
        self.superposition = superposition
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc_out = nn.ModuleList()
        for _ in range(n_tasks):
            self.fc_out.append(nn.Linear(hidden2, num_classes)) 

        if superposition:
            self.context1 = torch.randint(0, 2, (n_tasks, hidden1)) * 2 - 1
            self.context2 = torch.randint(0, 2, (n_tasks, hidden2)) * 2 - 1

    def forward(self, inputs, task_id=None, targets=None):

        flattened_inputs = inputs.reshape(inputs.size(0),-1)
        logits = F.relu(self.fc1(flattened_inputs))

        
        if self.superposition:
            #print (logits.shape)
            #print (self.context1.shape)
            logits = logits * self.context1[task_id]

        logits = F.relu(self.fc2(logits))

        if self.superposition:
            logits = logits * self.context2[task_id]


        logits = self.fc_out[task_id](logits)

        return logits
    

def train_model (model, train_loader, test_loader, batch_size, permutations, n_epochs=1, n_tasks = 5):

   
        


    model.train()


    criterion = nn.CrossEntropyLoss()

    train_loss_history = []

    for run in range(1):


        print ("_"*50)
        print (f"Run {run+1}")
        print ("_"*25)

        for t in range(n_tasks):

            optimizer = optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(n_epochs):
                running_loss = 0.0
                correct = 0
                total = 0

                for images, labels in train_loader:
                    images = images.to(device)   # (B, 1, 28, 28)
                    labels = labels.to(device)   # (B,)

                    print ("PCA ", apply_pca_to_batch(images))


                    # Flatten and permute pixels
                    B = images.size(0)
                    images = images.view(B, -1)         # (B, 784)
                    images = images[:, permutations[t]]            # (B, 784) permuted

                
                
                    logits = model(images, t)
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



            # Inside the task loop, after training:
            test_acc = evaluate(model, test_loader, permutations[0], 0)
            print(f"Task {1} | Test accuracy on its own permutation: {test_acc:.2f}%")
            
    model.eval()

    return model, train_loss_history




def apply_pca_to_batch(tensors, n_components=2):
    tensor_cpu = tensors.to('cpu')
    np_array = tensor_cpu.numpy()
    flat_array = np_array.reshape(np_array.shape[0], -1)  # batch size x features
    pca = PCA(n_components=n_components)
    return pca.fit_transform(flat_array)




def evaluate(model, loader, perm, task_id):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            B = images.size(0)
            images = images.view(B, -1)
            images = images[:, perm]

            logits = model(images, task_id)
            _, preds = logits.max(1)
            correct += preds.eq(labels).sum().item()
            total += B
    return correct / total * 100.0





        


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

print ("_"*100)
print ("NO SUPERPOSITION")

mlp1 = MLP(n_tasks=n_tasks).to(device)


mlp1, train_loss_history = train_model(model=mlp1, 
                                       train_loader=train_loader, 
                                       test_loader=test_loader, 
                                       permutations=permutations,
                                       batch_size=batch_size, 
                                       n_tasks = n_tasks)

print ("_"*50)
print (train_loss_history)

#print ("_"*50)


#for i in range(n_tasks):
    #evaluate(mlp1, test_loader, permutations, i)

print ("_"*50)

print ("SUPERPOSITION")

mlp2 = MLP(superposition=True, n_tasks=n_tasks).to(device)


mlp2, train_loss_history = train_model(model=mlp2, 
                                       train_loader=train_loader, 
                                       test_loader=test_loader, 
                                       permutations=permutations,
                                       batch_size=batch_size,
                                       n_tasks = n_tasks)

print ("_"*50)
print (train_loss_history)

#print ("_"*50)


#for i in range(n_tasks):
#    evaluate(mlp2, test_loader, permutations, i)

print ("_"*100)



#for epochs in range(n_epochs):

#    for image, label in train_loader:
#        batch_logits = mlp.forward(image)
#        predicted_classes = torch.argmax(batch_logits, dim=1)
#        print (predicted_classes)
   

   
    

#print (len(loader))

#print (loader[1])




