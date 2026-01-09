import matplotlib.pyplot as plt

from utilities import run_experiment




n_tasks = 5


#run with mnist, cifar, perm, diff rotation angles. with and without superposition. display all acc histories

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
