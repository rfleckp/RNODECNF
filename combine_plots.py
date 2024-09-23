import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from plots import set_plot_limits
from models import MLP, NODE
from data import sampler

# List of model paths and their respective datasets
training_types = ['otcfm', 'rnode']
name = 100_000
datasets = ['moons', 'gaussians', 'spirals', 'checkerboard']
model_paths = [
    [f'{dataset}/{training}/models/{name}_model.pt' 
    for training in training_types]
    for dataset in datasets]

# Setup device and random seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 15
samples = 3000
rows = len(datasets)
columns = 3
torch.manual_seed(seed)
np.random.seed(seed)

# Time points
t = torch.linspace(1, 0, 100).type(torch.float32)

# Colormap and normalization
cmap = plt.get_cmap('viridis_r')
norm = Normalize(vmin=0, vmax=1)

# Create a figure with subplots: 2 rows (1st for models, 2nd for datasets)
fig, axes = plt.subplots(rows, 3, figsize=(12,16))  # Adjust figure size as needed

for row in range(rows):

    for idx in range(columns):
        if idx < 2:
            # Load model
            model_path = model_paths[row][idx]
            odefunc = MLP().to(device)
            odefunc.load_state_dict(torch.load(model_path, map_location=device))
            odefunc.eval()
            
            # Create the flow for the current model
            x = torch.randn((samples if row<2 else 20000,2))
            cont_NF = NODE(odefunc)
            flow = cont_NF(x, traj=True, t=t).detach().cpu().numpy()
            
            first_color = cmap(norm(0))
            last_color = cmap(norm(1))  
            
            ax = axes[row, idx]  
            
            with torch.no_grad():
                for i in range(samples):
                    points = flow[:, i, :2]
                    segments = np.array([points[j:j+2] for j in range(len(points)-1)])
                    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=.1, zorder=1)
                    lc.set_array(t[:-1].numpy())
                    ax.add_collection(lc)
                
                # Scatter plot for base and target
                ax.scatter(flow[0, :, 0], flow[0, :, 1], color=last_color, s=.5, label="base", zorder=0)
                ax.scatter(flow[-1, :, 0], flow[-1, :, 1], color=first_color, s=.5, label="target", zorder=2)

        else:
            target = sampler(datasets[row])(samples if row<2 else 20000)
            first_color = cmap(norm(0))
            ax = axes[row, idx]
            ax.scatter(target[:, 0], target[:, 1], color=first_color, s=.5)

        # Remove ticks for model plots
        limits = set_plot_limits(datasets[row])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([-limits, limits])
        ax.set_ylim([-3.5 if datasets[row]=='moons' else -limits, 3.5 if datasets[row]=='moons' else limits])

        # Add legend for the first subplot in the first row
        if row == 0:
            """handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())"""
            if idx == 0:
                ax.set_title(f"OT-FM")
            elif idx == 1:
                ax.set_title(f"RNODE")
            else:
                ax.set_title(f"Ground Truth")


# Show the figure with both rows of plots
plt.tight_layout()
plt.savefig('combined_plots')
