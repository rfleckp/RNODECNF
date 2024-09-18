import os
import torch
import argparse
import numpy as np
from models import NODE
import matplotlib.pyplot as plt
from test import setup_model_and_data


#TOY PLOTS

def set_plot_limits(dataset: str):
    if dataset == 'moons':
        return 6
    elif dataset == 'spirals':
        return 14
    elif dataset == 'gaussians':
        return 5
    elif dataset == 'checkerboard':
        return 4
    elif dataset == 'normal':
        return 25

def toy_density_estimation1(model_path:str, seed:int = 3, samples:int =20000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, training, _, name = model_path.split('/')
    directory = os.path.join(dataset, training, 'plots')
    os.makedirs(directory, exist_ok=True)

    x, y, odefunc = setup_model_and_data(dataset, seed, samples)
    t=torch.linspace(1,0,4).type(torch.float32)
    x = x[0]

    odefunc.load_state_dict(torch.load(model_path, map_location=device))
    odefunc.eval()

    cont_NF = NODE(odefunc)
    flow = cont_NF(x, traj=True, t=t)
    limits = set_plot_limits(dataset)
    
    with torch.no_grad():
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        for i, ax in enumerate(axes):
            #ax.set_facecolor('black')
            ax.scatter(flow[i,:,0], flow[i,:,1], color='royalblue', s=0.2, alpha=.05)
            ax.set_title(f't = {t[i]:.2f}')

            ax.set_xlim(-limits, limits)  
            ax.set_ylim(-limits, limits)

            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(directory, f'{dataset}_density1'))
        #plt.cla()


import seaborn as sns

def toy_density_estimation2(model_path: str, seed: int=3, steps: int=4, samples: int=20000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, training, _, name = model_path.split('/')
    directory = os.path.join(dataset, training, 'plots')
    os.makedirs(directory, exist_ok=True)

    x, y, odefunc = setup_model_and_data(dataset, seed, samples)
    t=torch.linspace(1,0,steps).type(torch.float32)
    x = x[0]

    odefunc.load_state_dict(torch.load(model_path, map_location=device))
    odefunc.eval()

    cont_NF = NODE(odefunc)
    flow = cont_NF(x, traj=True, t=t)
    limits = set_plot_limits(dataset)    
    
    with torch.no_grad():
        fig, axes = plt.subplots(1, steps, figsize=((steps)*4*1.5, 4), dpi=300)

        for i, ax in enumerate(axes):
            ax.set_facecolor('black')
            ax.set_xticks([])
            ax.set_yticks([])          

            sns.kdeplot(x=flow[i,:,0], y=flow[i,:,1], fill=True, thresh=0, levels=200, cmap='magma', ax=ax, clip=(limits,-limits), bw_adjust=.3)
            ax.set_title(f't = {t[i]:.2f}')

        plt.tight_layout()
        plt.savefig(os.path.join(directory, f'{dataset}_density2'))
        plt.cla()

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
def plot_toy_flow1(model_path: str, samples: int=3000, seed: int=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, training, _, name = model_path.split('/')
    directory = os.path.join(dataset, training, 'plots')
    os.makedirs(directory, exist_ok=True)

    x, y, odefunc = setup_model_and_data(dataset, seed, samples)
    t = torch.linspace(1, 0, 100).type(torch.float32)
    x = x[0]

    odefunc.load_state_dict(torch.load(model_path, map_location=device))
    odefunc.eval()
    cont_NF = NODE(odefunc)
    flow = cont_NF(x, traj=True, t=t).detach().cpu().numpy()

    cmap = plt.get_cmap('viridis_r')
    norm = Normalize(vmin=0, vmax=1)
    first_color = cmap(norm(0))
    last_color = cmap(norm(1))  

    with torch.no_grad():
        fig, ax = plt.subplots()
        for i in range(samples):
            points = flow[:, i, :2]
            segments = np.array([points[j:j+2] for j in range(len(points)-1)])
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=.1, zorder=1)
            lc.set_array(t[:-1].numpy())
            ax.add_collection(lc)

        ax.scatter(flow[0, :, 0], flow[0, :, 1], color=last_color, s=.5, label="base", zorder=0)
        ax.scatter(flow[-1, :, 0], flow[-1, :, 1], color=first_color, s=.5, label="target", zorder=2)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(os.path.join(directory, f'{dataset}_flow1'))
    plt.clf()


def plot_toy_flow2(model_path: str, samples: int=3000,  seed: int=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, training, _, name = model_path.split('/')
    directory = os.path.join(dataset, training, 'plots')
    os.makedirs(directory, exist_ok=True)

    x, y, odefunc = setup_model_and_data(dataset, seed, samples)
    t=torch.linspace(1,0,100).type(torch.float32)
    x = x[0]

    odefunc.load_state_dict(torch.load(model_path, map_location=device))
    odefunc.eval()
    cont_NF = NODE(odefunc)
    flow = cont_NF(x, traj=True, t=t).detach().cpu().numpy()

    with torch.no_grad():
        fig, ax = plt.subplots()

        ax.plot(flow[:,:,0], flow[:,:,1], color="royalblue", alpha=.05, zorder=0, label="flow")            
        ax.scatter(flow[0,:,0],flow[0,:,1], c="royalblue", s=1, label="base", zorder=1)
        ax.scatter(flow[-1,:,0], flow[-1,:,1], c="royalblue", s=1, label="target", zorder=1)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(os.path.join(directory, f'{dataset}_flow2'))
    plt.clf()


#MNIST PLOTS

def generate_grid(model_path: str, seed: int=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, training, _, name = model_path.split('/')
    directory = os.path.join(dataset, training, 'plots')
    os.makedirs(directory, exist_ok=True)

    x, _, odefunc = setup_model_and_data(dataset, seed)
    x = x[0, :25, :, :]

    t=torch.linspace(1,0,2).type(torch.float32)

    odefunc.load_state_dict(torch.load(model_path, map_location=device))
    odefunc.eval()

    cont_NF = NODE(odefunc)
    images = cont_NF(torch.randn((25,1,28,28)), traj=False, t=t).detach().cpu().numpy()

    fig, axes = plt.subplots(5, 5, figsize=(10, 10))  
    axes = axes.flatten()  

    for i in range(25):
        ax = axes[i]
        ax.imshow(images[i, 0, :, :], cmap='gray')  
        ax.axis('off')  

    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'{dataset}_grid{name.split('_')[0]}'))
    plt.cla()


def plot_mnist_flow(model_path: str, seed: int=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, training, _, name = model_path.split('/')
    directory = os.path.join(dataset, training, 'plots')
    os.makedirs(directory, exist_ok=True)

    x, _, odefunc = setup_model_and_data(dataset, seed)
    x = x[0, :1, :, :]

    t=torch.linspace(1,0,5).type(torch.float32)

    odefunc.load_state_dict(torch.load(model_path, map_location=device))
    odefunc.eval()

    cont_NF = NODE(odefunc)
    flow = cont_NF(x, traj=True, t=t).detach().cpu().numpy()
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    for i, ax in enumerate(axes):
        ax.imshow(flow[i, 0, 0, :, :], cmap='gray')
        ax.set_title(f't = {t[i]:.2f}')
        ax.axis('off')  

    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'{dataset}_flow{name.split('_')[0]}'))
    plt.cla()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str) 
    parser.add_argument("--dataset", type=str, choices=['mnist', 'toy'])
    parser.add_argument("--flow", default = False, type=bool)
    parser.add_argument("--density", default = False, type=bool)

    args = parser.parse_args()
    
    if args.dataset == 'mnist':
        for i in os.listdir(args.model_path):
            generate_grid(os.path.join(args.model_path, i))
            plot_mnist_flow(os.path.join(args.model_path, i))
    else: 
        if args.density:
            toy_density_estimation1(args.model_path, samples=200_000)
            toy_density_estimation2(args.model_path, samples=20_000)
        if args.flow:
            plot_toy_flow1(args.model_path, samples=3_000)
            plot_toy_flow2(args.model_path, samples=1_000)
