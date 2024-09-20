import torch
import os
import time
import numpy as np
import ot as pot
from tabulate import tabulate
from data import sampler, mnist_test_loader
from models import MLP, UNet, NODE


def flow_length(flow):
    diff = flow[1:] - flow[:-1]
    if diff.dim() == 3:
        dist = torch.linalg.norm(diff, dim=-1)
        straight = torch.linalg.norm(flow[-1]-flow[0], dim=1)
    else:
        dist = torch.linalg.norm(diff, dim=(-2,-1)).squeeze()
        straight = torch.linalg.norm(flow[-1]-flow[0], dim=(-2,-1)).squeeze()
    trajectory_length = dist.sum(dim=0)
    return torch.mean(trajectory_length-straight)


def flow_straightness(odefunc, flow):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = flow.size(0)
    t = torch.linspace(1, 0, steps).to(device)
    straight_vec = torch.linalg.norm(flow[-1]-flow[0], dim=1)**2
    norm = 0

    with torch.no_grad():
        for i in range(steps):
            norm += torch.linalg.norm(odefunc(t[i], flow[i]), dim=1)**2
        
    return torch.mean(norm/steps-straight_vec)


import torch.nn as nn
from torchdiffeq import odeint as odeint
class Path_Energy(nn.Module):
    def __init__(self, ode_func: nn.Module, dataset: str):
        super().__init__()
        self.odefunc = ode_func
        self.nfe = 0
        if dataset == 'mnist':
            self.mnist = True
        else:
            self.mnist = False

    def forward(self, t, states):
        z = states[0]                                           
        dz_dt = self.odefunc(t, z)
        self.nfe += 1
        if self.mnist:
            dE_dt = (torch.linalg.matrix_norm(dz_dt))**2
        else:
            dE_dt = (torch.linalg.vector_norm(dz_dt, dim=1, keepdims=True)**2)                       

        return (dz_dt, dE_dt)

def normalized_path_energy(odefunc, x, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #initial_values = (x, torch.zeros((x.size(0),1)).to(device))
    if dataset == 'mnist':
        E = torch.linalg.matrix_norm(odefunc(torch.tensor(0).to(device), x))
    else:
        E = torch.linalg.norm(odefunc(torch.tensor(0).to(device), x), dim=1)
    initial_values = (x, E)
    augmented_dynamics = Path_Energy(odefunc, dataset)
    flow, energy = odeint(augmented_dynamics, initial_values, 
                          t=torch.linspace(1,0,100).to(device), 
                          method='dopri5',
                          atol=1e-4,
                          rtol=1e-4)
    w2d = wasserstein2(flow[0],flow[-1])**2
    E = torch.abs(energy[-1].mean())
    
    return flow, torch.abs(E-w2d)/w2d, augmented_dynamics.nfe


def wasserstein2(x, y):
    bs = x.size(0)
    a, b = pot.unif(bs), pot.unif(bs)

    if x.dim() > 2:
        x = x.reshape(bs, -1)
        y = y.reshape(bs, -1)
    M = torch.cdist(x, y)**2
    dist = pot.emd2(a, b, M.detach().cpu().numpy(), numItermax=int(1e7))

    return dist**.5


def save_logs(path, data, train, first_write=False, params=None):
    os.makedirs(path + "/logs", exist_ok=True)

    if train:
        file_path = path + "/logs/train.txt"
        if "mnist" in path:
            headers = ["epoch", "   loss    ", "    time    "]
        else:
            headers = ["batch", "   loss    ", "    time   "]
    else:
        file_path = path + "/logs/test.txt"
        if "mnist" in file_path:
            headers = ["    epoch   ", "n_logl", "straight", "w2dist", "energy", "nfe"]
        else:
            headers = ["    batch   ", "n_logl", "straight", "w2dist", "energy", "nfe"]      

    table = tabulate(data, tablefmt="grid", floatfmt=".3f") 
    mode = "w" if first_write else "a"  
    with open(file_path, mode) as f:
        if first_write:
            if train:
                f.write("Parameters:\n")
                for param, value in params.items():
                    f.write(f"  {param}: {value}\n")
                f.write("\n\n")
            f.write("\t".join(headers) + "\n")  
        f.write(table + "\n")

def format_elapsed_time(elapsed_seconds):
    hours = int(elapsed_seconds // 3600) 
    minutes = int((elapsed_seconds % 3600) // 60)  
    seconds = int(elapsed_seconds % 60)
    
    formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"
    return formatted_time

def setup_model_and_data(dataset: str, seed: int=10, toy_samples: int=2000):
    """
    for MNIST: returns batched samples from the normal distribution matching the test-loader
        

    for TOY: returns 2000 unsqueezed samples from both the normal and target distribution
        x[0], y[0][0] to access samples 2000x2
            
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if dataset == "mnist":
        batch_size = 128
        y = mnist_test_loader(batch_size=batch_size)
        x = torch.randn((len(y), batch_size, 1, 28, 28))
        odefunc = UNet().to(device)
    else:
        y = sampler(dataset)(toy_samples)
        x = torch.randn_like(y).unsqueeze(0).to(device)
        y = [(y.to(device), 0)]
        odefunc = MLP().to(device)
    
    return x, y, odefunc

    
def evaluate_model(path, x, y, odefunc, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    odefunc.load_state_dict(torch.load(path, map_location=device))
    odefunc.eval()

    cont_NF = NODE(odefunc).to(device)
    c, neg_log, length, w2d, path_energy, nfe, nll = 0, 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for y_batched, _ in y:
            x_batched = x[c].to(device)
            y_batched = y_batched.to(device)
            c += 1
            if x_batched.shape[0] != y_batched.shape[0]:
                x_batched = x_batched[:y_batched.shape[0]]
            flow, energy, fe = normalized_path_energy(odefunc, x_batched, dataset)
            path_energy += energy
            nfe += fe
            length += flow_length(flow)
            #straight += flow_straightness(odefunc, flow)
            neg_log = cont_NF.log_likelihood(y_batched, 5)
            nll += neg_log
            w2d += wasserstein2(flow[-1], y_batched)
    del odefunc
    return [[nll/c, length/c, w2d/c, path_energy/c, nfe/c]]

#prints a progress bar
def progress_bar(i, total, time, est_time):
    progress = (i) / total
    bar_length = 40
    block = int(round(bar_length * progress))
    progress_bar = "#" * block + "-" * (bar_length - block)
    print(f"\rProgress: [{progress_bar}] {int(progress * 100)}%, Time: {time}<{est_time}", end="")

def extract_batch_number(filename):
    return int(filename.split('_')[0])

def evaluate_models(directory: str):
    dataset, training, _ = directory.split('/')
    print(f"\n\nTEST {training} for {dataset}")
    path = os.path.join(dataset, training)
    x, y, odefunc = setup_model_and_data(dataset)
    model_list = os.listdir(directory)

    try:
        model_list.remove('.DS_Store')
    except:   
        pass

    sorted_files = sorted(model_list, key=extract_batch_number)
    total = len(sorted_files)
    first = True
    start = time.time()
    i = 0
    #existing_data = extract_data(os.path.join(path,'logs/test.txt'))

    for model in sorted_files:
        model_path = os.path.join(directory, model)
        data = evaluate_model(model_path, x, y, odefunc, dataset)
        data[0].insert(0, model)
        #existing_data[i+1][3] = data[0][3]
        #data = [existing_data[i+1]]
        save_logs(path, data, train=False, first_write=first)
        first = False

        if dataset != 'mnist':
            i += 1
            elapsed_time = time.time()-start
            estimated_time = ((elapsed_time)/i)*total
            progress_bar(i, total, format_elapsed_time(elapsed_time), format_elapsed_time(estimated_time))
        
        del data


def evaluate_dataset(dataset):
    training_methods = os.listdir(dataset)
    try:
        training_methods.remove('.DS_Store')
    except:   
        pass

    for i in training_methods:
        directory = os.path.join(dataset, i, 'models')
        evaluate_models(directory)

