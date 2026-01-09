from utilities import run_experiment
import torch
from itertools import product

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from datetime import datetime
import os

def plot_combined_losses(results_dict, rotation_angles=[None, 10, 15, 20, 36], save_dir='plots'):
    """
    Superposition → linestyle (True=solid, False=dashed)
    Angle → color + marker (auto-cycling)
    First history=MNIST, second=CIFAR per condition
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Maps
    linestyle_map = {True: '-', False: '--'}
    
    # Auto-color + marker cycle for angles
    colors = plt.cm.viridis(np.linspace(0, 1, len(rotation_angles)))
    markers = ['o', 's', '^', 'D', 'v', 'p', 'P', '*'][:len(rotation_angles)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Track for legend
    legend_elements = []
    plotted_conditions = set()

    for (superpos, angle), histories in results_dict.items():
        if angle not in rotation_angles: continue
            
        mnist_losses, cifar_losses = histories
        cond_name = f'S={superpos}, A={angle or "None"}'
        ls = linestyle_map[superpos]
        
        # Get color/marker by angle index
        angle_idx = rotation_angles.index(angle)
        color = colors[angle_idx]
        marker = markers[angle_idx]

        # MNIST (first task)
        line1, = ax1.plot(mnist_losses, ls=ls, color=color, marker=marker, 
                         linewidth=2.5, markersize=6, label=cond_name)
        
        # CIFAR (second task)  
        line2, = ax2.plot(cifar_losses, ls=ls, color=color, marker=marker, 
                         linewidth=2.5, markersize=6)

        # Legend only once per condition
        if cond_name not in plotted_conditions:
            legend_elements.append(line1)
            plotted_conditions.add(cond_name)

    # Axes
    ax1.set_title('MNIST')
    ax1.set_xlabel('Task number')
    ax1.set_ylabel('Accuracy of a model on the original task')
    ax1.grid(True, alpha=0.3)
    ax1.legend(handles=legend_elements, loc='upper right')
    
    ax2.set_title('CIFAR')
    ax2.set_xlabel('Task number')
    ax2.set_ylabel('Accuracy of a model on the original task')
    ax2.grid(True, alpha=0.3)
    ax2.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    
    filename = f'{save_dir}/combined_losses_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'Saved: {filename}')
    plt.show()

'''

def plot_combined_losses(results_dict, save_dir='plots'):
    """
    Superposition → linestyle (True=solid, False=dashed)
    Angle → color (None=blue, 15=orange, 45=red)
    First history=MNIST, second=CIFAR per condition
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Maps
    linestyle_map = {True: '-', False: '--'}
    color_map = {None: 'blue', 15: 'orange', 45: 'red'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Track for legend
    mnist_lines = []
    cifar_lines = []

    for (superpos, angle), histories in results_dict.items():
        mnist_losses, cifar_losses = histories
        cond_name = f'S={superpos}, A={angle or "None"}'
        ls = linestyle_map[superpos]
        color = color_map.get(angle, 'gray')

        # MNIST (circles)
        line1, = ax1.plot(mnist_losses, ls=ls, color=color, marker='o', 
                          linewidth=2.5, markersize=6)
        mnist_lines.append(line1)
        
        # CIFAR (squares)
        line2, = ax2.plot(cifar_losses, ls=ls, color=color, marker='s', 
                          linewidth=2.5, markersize=6)
        cifar_lines.append(line2)

    # Axes
    ax1.set_title('MNIST')
    ax1.set_xlabel('Task number')
    ax1.set_ylabel('Accuracy of a model on the original task')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('CIFAR')
    ax2.set_xlabel('Task number')
    ax2.set_ylabel('Accuracy of a model on the original task')
    ax2.grid(True, alpha=0.3)

    # Custom legend with solid/dashed distinction
    legend_elements = [
        mlines.Line2D([], [], color='blue', ls='-', marker='o', label='S=True, A=None'),
        mlines.Line2D([], [], color='orange', ls='-', marker='o', label='S=True, A=15'), 
        mlines.Line2D([], [], color='red', ls='-', marker='o', label='S=True, A=45'),
        mlines.Line2D([], [], color='blue', ls='--', marker='o', label='S=False, A=None'),
        mlines.Line2D([], [], color='orange', ls='--', marker='o', label='S=False, A=15'),
        mlines.Line2D([], [], color='red', ls='--', marker='o', label='S=False, A=45'),
    ]
    
    ax1.legend(handles=legend_elements, loc='upper right')
    ax2.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    
    filename = f'{save_dir}/combined_losses_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'Saved: {filename}')
    plt.show()
    return fig
'''

print("cuda is available: ", torch.cuda.is_available())


n_tasks = 10


#run with mnist, cifar, perm, diff rotation angles. with and without superposition. display all acc histories

#datasets = ["MNIST", "CIFAR"]

superposition_options = [True, False]

#rotation angle of None means a permutation instead of rotation
rotation_angles = [None, 10, 15, 20, 36]
#rotation_angles = [None, 15]



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



plot_combined_losses(accuracy_histories_of_experimental_conditions)




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