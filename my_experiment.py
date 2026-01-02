import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset



import numpy as np
import matplotlib.pyplot as plt

from utilities import MLP, train_model, evaluate


fig, axes = plt.subplots(1, 2, figsize=(15, 6))


train_set = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)



pca_instances_loader = get_PCA_instances_loader(train_set=train_set)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print ("_"*100)
print ("NO SUPERPOSITION")

mlp1 = MLP(n_tasks=n_tasks).to(device)


mlp1, train_loss_history, penultimate_logits = train_model(model=mlp1, 
                                       train_loader=train_loader, 
                                       test_loader=test_loader, 
                                       permutations=permutations,
                                       pca_instances_loader=pca_instances_loader,
                                       batch_size=batch_size, 
                                       n_tasks = n_tasks)


print ("_"*50)
print (train_loss_history)


print ("_"*50)

print (penultimate_logits)

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


print ("SUPERPOSITION")

mlp2 = MLP(superposition=True, n_tasks=n_tasks).to(device)


mlp2, train_loss_history, penultimate_logits = train_model(model=mlp2, 
                                       train_loader=train_loader, 
                                       test_loader=test_loader, 
                                       permutations=permutations,
                                       pca_instances_loader=pca_instances_loader,
                                       batch_size=batch_size,
                                       n_tasks = n_tasks)


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




print ("_"*50)
print (train_loss_history)


print ("_"*100)

