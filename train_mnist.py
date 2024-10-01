import torch
import time
from models import create_model, create_regularization_fns
import numpy as np
import os
import argparse

import torch.optim as optim
from training_utils import ExactOptimalTransportConditionalFlowMatcher, compute_bits_per_dim, standard_normal_logprob
from torchvision.utils import save_image
from test import save_logs, format_elapsed_time
from data import mnist_train_loader
from plots import generate_grid

def update_lr(optimizer, itr, rate):
    iter_frac = min(float(itr + 1) / max(1000, 1), 1.0)
    lr = rate * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def add_noise(x, nbits=8):
    noise = x.new().resize_as_(x).uniform_()
    return x.add_(noise).div_(2**nbits)

def shift(x, nbits=8):
    if nbits<8:
        x = x // (2**(8-nbits))

    return x.add_(1/2).div_(2**nbits)

def unshift(x, nbits=8):
    return x.add_(-1/(2**(nbits+1)))

def compute_bits_per_dim(x, model):
    nvals = 2**8
    zero = torch.zeros(x.shape[0], 1).to(x)

    z, delta_logp, reg_states = model(x, zero)  # run model forward

    reg_states = tuple(torch.mean(rs) for rs in reg_states)

    logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp

    logpx_per_dim = torch.sum(logpx) / x.nelement()  # averaged over batches
    bits_per_dim = -(logpx_per_dim - np.log(nvals)) / np.log(2)

    return bits_per_dim, (x, z), reg_states

def train_mnist_rnode(params):

    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    regularization_fns, regularization_coeffs = create_regularization_fns()
    model = create_model(regularization_fns).to(device)
    first = True

    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=0)
    train_loader = mnist_train_loader(params["batch_size"])

    path = "mnist/rnode2"
    os.makedirs(path + "/models", exist_ok=True)
    start = time.time()
    itr = 0
    model.train()
    fixed_z = cvt(torch.randn(25, *(1,28,28)))


    for epoch in range(1, params['n_epochs']+1):
        epoch_loss = 0.0
        progress_bar = enumerate(train_loader)
        num = 0
        for i, (samples, labels) in progress_bar:
            optimizer.zero_grad()
            update_lr(optimizer, itr, params['learning_rate'])
            optimizer.zero_grad()

            # cast data and move to device+
            x = add_noise(cvt(samples), nbits=8)
            
            # compute loss
            bpd, (x, z), reg_states = compute_bits_per_dim(x, model)
            #print('bpd: ', bpd)
            if np.isnan(bpd.data.item()):
                raise ValueError('model returned nan during training')
            elif np.isinf(bpd.data.item()):
                raise ValueError('model returned inf during training')
            
            loss = bpd
            if regularization_coeffs:
                reg_loss = sum(
                    reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
                )
                loss = loss + reg_loss
            #print('loss computed')
            loss.backward()
            #print('backward fulfilled')
            optimizer.step()
            epoch_loss += loss
            num += 1

            elapsed_time = format_elapsed_time(time.time()-start)
            data = [[epoch, epoch_loss/num, elapsed_time]]
            save_logs(path, data, train=True, params=params, first_write=first)
            first = False

        torch.save(model.state_dict(), os.path.join(path + "/models", f"{epoch}_model.pt"))
        generated_samples, _, _ = model(cvt(torch.randn(25, *(1,28,28))), reverse=True)
        generated_samples = generated_samples.view(-1, *(1,28,28))
        nb = int(np.ceil(np.sqrt(float(fixed_z.size(0)))))
        fig_filename = os.path.join(path, 'plots')
        os.makedirs(fig_filename, exist_ok=True)
        fig_filename = os.path.join(fig_filename, f'{epoch}-grid.png')
        save_image(unshift(generated_samples, nbits=8), fig_filename, nrow=nb)

        """if epoch%5==0:
            torch.save(model.state_dict(), os.path.join(path + "/models", f"{epoch}_model.pt"))
            elapsed_time = format_elapsed_time(time.time()-start)
            data = [[epoch, epoch_loss/num, elapsed_time]]
            save_logs(path, data, train=True, params=params, first_write=first)
            first = False"""



def train_mnist_cfm(params):

    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)
    model = create_model().to(device)
    first = True

    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=0)
    train_loader = mnist_train_loader(params["batch_size"])

    path = "mnist/otcfm2"
    os.makedirs(path + "/models", exist_ok=True)
    start = time.time()
    first = True
    model.train()
    fixed_z = cvt(torch.randn(25, *(1,28,28)))
    t = cvt(torch.randn(25))
    t = t.view(t.shape[0],1,1,1)
    print(t.shape)
    #print(model.transforms[0].chain[1].odefunc.diffeq(t, fixed_z).shape)
    print(model.transforms[0])
    #for idx in range(len(model.transforms)):
            #print(idx, model.transforms[idx])
    '''
    for epoch in range(1, params['n_epochs']+1):
        epoch_loss = 0.0
        progress_bar = enumerate(train_loader)
        num = 0
        for i, (samples, labels) in progress_bar:
            optimizer.zero_grad()

            x0 = torch.randn_like(samples).to(device)
            t, xt, ut = FM.sample_location_and_conditional_flow(samples.to(device), x0)
            vt = model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            num += 1
        
        if epoch%5==0:
            torch.save(model.state_dict(), os.path.join(path + "/models", f"{epoch}_model.pt"))
            elapsed_time = format_elapsed_time(time.time()-start)
            data = [[epoch, epoch_loss/num, elapsed_time]]
            save_logs(path, data, train=True, params=params, first_write=first)
            first = False'''



def main(n_epochs = 100,          
        batch_size = 200,
        odeint_method = 'rk4',
        learning_rate = 1e-3,
        lambda_k = .01,       
        lambda_j = .01,       
        sigma = 0.001):         
    """
    Parameters for Training
        dataset: ['moons', 'gaussians', 'circles', 'mnist']
        training: ['node', 'rnode', 'cfm']

    MNIST datset parameters
        n_epochs: number of epochs for training    

    TOY dataset parameters
        n_batches: number of batches for training
        batch_size: size of training batches

    RNODE parameters
        odeint_method: ['euler', 'rk4', 'dopri5', 'dopri8']
        lambda_k: regularization constant for the kinetic energy
        lambda_j: regularization constant for the frobenius norm

    CFM parameters
        sigma: variance of gaussian probability paths
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-method", default="rnode",
                        choices=["rnode", "otcfm"])
    args = parser.parse_args()

    params = {
        "seed": 22,
        "learning_rate": learning_rate,
    }    

    params["n_epochs"] = n_epochs
    params["batch_size"] = batch_size

    if args.training_method == "rnode":
        params["odeint_method"] = odeint_method
        params["lambda_k"] = lambda_k
        params["lambda_j"] = lambda_j
        train_mnist_rnode(params)

    elif args.training_method == "otcfm":
        params["sigma"] = sigma
        train_mnist_cfm(params)
    

if __name__ == "__main__":
    main()