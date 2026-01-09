import matplotlib.pyplot as plt

from utilities import run_experiment

from itertools import product



n_tasks = 2


#run with mnist, cifar, perm, diff rotation angles. with and without superposition. display all acc histories

#datasets = ["MNIST", "CIFAR"]

superposition_options = [True, False]

#rotation angle of None means a permutation instead of rotation
rotation_angles = [None, 15, 45]

experimental_conditions = list(product(superposition_options, rotation_angles))

#print (experimental_conditions)

accuracy_histories_of_experimental_conditions = {}

for experimental_condition in experimental_conditions:
    superposition = experimental_condition[0]
    rotation_angle = experimental_condition[1] 
 
    print ("_"*100)
    print (f"Current experimental condition.")
    print ("_"*75)
    print (f"Superposition: {superposition}")
    print (f"Rotation angle: {rotation_angle}")
    

    mnist_acc_hist = run_experiment(dataset_name = "MNIST",
                image_rotation_angle_per_task = rotation_angle,
                n_tasks=n_tasks,
                superposition = superposition
                )
    print ("_"*50)
    print (f"Current MNIST acc hist: {mnist_acc_hist}")

    cifar_acc_hist = run_experiment(dataset_name = "CIFAR",
                    image_rotation_angle_per_task = rotation_angle,
                    n_tasks=n_tasks,
                    superposition = superposition
                    )
    print ("_"*50)
    print (f"Current CIFAR acc hist: {cifar_acc_hist}")
    
    accuracy_histories_of_experimental_conditions[experimental_condition] = [mnist_acc_hist, cifar_acc_hist]

print (accuracy_histories_of_experimental_conditions)



"""
run_experiment(dataset_name = "CIFAR",
                image_rotation_angle_per_task = 10,
                n_tasks=n_tasks,
                superposition = False
                )

run_experiment(dataset_name = "CIFAR",
                image_rotation_angle_per_task = 10,
                n_tasks=n_tasks,
                superposition = True
                )
"""