import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as T
import torch.optim as optim
import torch
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

    def forward(self, inputs, task_id=None, get_penultimate_logits = False, targets=None):

        flattened_inputs = inputs.reshape(inputs.size(0),-1)
        logits = F.relu(self.fc1(flattened_inputs))

        
        if self.superposition:
            #print (logits.shape)
            #print (self.context1.shape)
            logits = logits * self.context1[task_id]

        logits = F.relu(self.fc2(logits))

        if self.superposition:
            logits = logits * self.context2[task_id]

        if not get_penultimate_logits:

            logits = self.fc_out[task_id](logits)

            return logits
        
        else:

            #logits.copy()?
            penultimate_logits = logits

            logits = self.fc_out[task_id](logits)

            return logits, penultimate_logits


    

def train_model (model,
                 train_loader,
                 test_loader,
                 permutations,
                 pca_instances_loader=None,
                 n_epochs=3,
                 n_tasks = 5,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):

   
    model.train()


    criterion = nn.CrossEntropyLoss()

    train_loss_history = []

    batch_penultimate_infos = []

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

                    #print ("PCA ", apply_pca_to_batch(images))


                    # Flatten and permute pixels
                    B = images.size(0)
                    #images = images.view(B, -1)
                                        
                                        
                    if permutations is not None:     
                        images = images.view(B, -1)         # (B, 784)
                        images = images[:, permutations[t]]     
                    else:
                        images = T.rotate(images, angle = 10*t)  
                        images = images.view(B, -1)                                                 # (B, 784) permuted

                
                
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

            
            

            if t==0 or t==5 or t==9:

                """
                for images, labels in pca_instances_loader:
                    _, batch_logits_penultimate = model.forward(inputs = images, 
                                                                task_id = t,
                                                                get_penultimate_logits = True)
                    
                    batch_logits_penultimate_pca = apply_pca_to_batch(batch_logits_penultimate)


                    batch_penultimate_infos.append((batch_logits_penultimate_pca, labels))
                """

                with torch.no_grad():
                    for images, true_labels in test_loader:  # true task labels
                        images = images.to(device)
                        B = images.size(0)
                        #images_flattened = images.view(B, -1)
                        if permutations is not None:
                            images_flattened = images.view(B, -1)
                            images_perm = images_flattened[:, permutations[t]]
                        else:
                            
                            images_perm = T.rotate(images, angle = 360*t/n_tasks)
                            images_perm = images_perm.view(B, -1)
                            #images_perm = images_flattened

                        #images_perm = images.view(B, -1)[:, permutations[t]]  # Task t permutation!
                        
                        _, penultimate = model(images_perm, t, get_penultimate_logits=True)
                        penult_pca = apply_pca_to_batch(penultimate)  # [B, 2]
                        
                        #batch_penultimate_infos.append((penult_pca.cpu().numpy(), true_labels.cpu().numpy()))
                        batch_penultimate_infos.append((penult_pca, true_labels.cpu().numpy()))
            #elif t==5:
            
            #elif t==9:


            # Inside the task loop, after training:
            if permutations is not None:
                test_acc = evaluate(model=model, loader=test_loader, task_id=0, perm=permutations[0])
            else:
                #test_acc = evaluate(model, test_loader, permutations[0], 0)
                test_acc = evaluate(model=model, loader=test_loader, task_id=0)
            print(f"Task {1} | Test accuracy on its own permutation: {test_acc:.2f}%")
            
    model.eval()

    return model, train_loss_history, batch_penultimate_infos


def apply_pca_to_batch(tensors, n_components=2):
    tensor_cpu = tensors.to('cpu')
    np_array = tensor_cpu.detach().numpy()
    flat_array = np_array.reshape(np_array.shape[0], -1)  # batch size x features
    pca = PCA(n_components=n_components)
    return pca.fit_transform(flat_array)








def evaluate(model, loader, task_id, perm = None, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            B = images.size(0)
            images = images.view(B, -1)

            if perm is not None:
                images = images[:, perm]

            logits = model(images, task_id)
            _, preds = logits.max(1)
            correct += preds.eq(labels).sum().item()
            total += B
    return correct / total * 100.0


#def run_experiment(train_loader,
#                   test_loader,
#                   permutations,
#                   input_dim = 784, 
#                   n_tasks = 10,
#                   superposition = False):

def run_experiment(#train_loader,
                   #test_loader,
                   dataset_name = "MNIST",
                   #input_dim = 784, 
                   batch_size = 128,
                   n_tasks = 10,
                   superposition = False):


    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_name.upper() == "MNIST":

        input_dim = 28 * 28 * 1
        hidden_dim = 128

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


        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

        permutations = torch.stack([torch.randperm(input_dim) for _ in range(n_tasks)])


    elif dataset_name.upper() == "CIFAR":

        input_dim = 32 * 32 * 3
        hidden_dim = 512

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 mean/std
        ])

        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


        permutations = None
    else:
        print ("Pass either MNIST or CIFAR as a dataset name for the run experiment method.")
        return


    print(f"Dataset: {dataset_name}, Train batches: {len(train_loader)}, Classes: {train_set.classes}")
    print ("_"*100)
    if superposition:
        print ("SUPERPOSITION")
    else:
        print ("NO SUPERPOSITION")

    mlp1 = MLP(superposition=superposition, n_tasks=n_tasks, input_dim=input_dim, hidden1=hidden_dim, hidden2=hidden_dim).to(device)


    mlp1, train_loss_history, penultimate_logits = train_model(model=mlp1, 
                                        train_loader=train_loader, 
                                        test_loader=test_loader, 
                                        permutations=permutations,
                                        n_tasks = n_tasks)


    print ("_"*50)
    print (train_loss_history)


    print ("_"*50)

    #print (penultimate_logits)

    points0 = penultimate_logits[0][0]  # shape: [num_samples, 2]
    labels0 = penultimate_logits[0][1]


    labels = np.array(labels0)

    plt.scatter(points0[labels0 == 0, 0], points0[labels0 == 0, 1], 
                c='red', marker='o', label='Label Task 1 0 (circle)')

    # Plot points with label 1 using square marker 's'
    plt.scatter(points0[labels0 == 1, 0], points0[labels0 == 1, 1], 
                c='red', marker='s', label='Label Task 1 1 (square)')




    points1 = penultimate_logits[1][0]  # shape: [num_samples, 2]
    labels1 = penultimate_logits[1][1]

    labels = np.array(labels1)

    plt.scatter(points1[labels1 == 0, 0], points1[labels1 == 0, 1], 
                c='blue', marker='o', label='Label 0 (circle)')

    # Plot points with label 1 using square marker 's'
    plt.scatter(points1[labels1 == 1, 0], points1[labels1 == 1, 1], 
                c='blue', marker='s', label='Label 1 (square)')


    points2 = penultimate_logits[2][0]  # shape: [num_samples, 2]
    labels2 = penultimate_logits[2][1]

    labels = np.array(labels2)

    plt.scatter(points2[labels2 == 0, 0], points2[labels2 == 0, 1], 
                c='green', marker='o', label='Label 0 Task 10 (circle)')

    # Plot points with label 1 using square marker 's'
    plt.scatter(points2[labels2 == 1, 0], points2[labels2 == 1, 1], 
                c='green', marker='s', label='Label 1 Task 10 (square)')





    #plt.scatter(penultimate_logits[0][0][:, 0], penultimate_logits[0][0][:, 1], c='red', label='Class 1')



    #plt.scatter(penultimate_logits[1][0][:, 0], penultimate_logits[1][0][:, 1], c='blue', label='Class 2')
    #plt.scatter(penultimate_logits[2][0][:, 0], penultimate_logits[2][0][:, 1], c='green', label='Class 3')

    plt.legend()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter plot of three arrays with different colors')
    plt.show()

