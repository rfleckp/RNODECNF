import torch
import os
import time
import numpy as np
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
from training_utils import ExactOptimalTransportConditionalFlowMatcher, aug_ode, reg_aug_mnist, reg_aug_toy, choose_timesteps, compute_bits_per_dim, standard_normal_logprob
from test import save_logs, format_elapsed_time, progress_bar
from data import sampler, mnist_train_loader
from models import MLP, UNet, Swish

def update_lr(optimizer, itr, rate):
    iter_frac = min(float(itr + 1) / max(1000, 1), 1.0)
    lr = rate * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def train_toy_node(target, params):
    
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)

    target_distr = sampler(target)
    initial_distr = torch.distributions.MultivariateNormal(torch.zeros(2).to(device), torch.eye(2).to(device))
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    model.train()

    batch_loss = 0
    start = time.time()
    first = True
    path = target + "/node"
    os.makedirs(path + "/models", exist_ok=True)

    for i in range(1, params['n_batches']+1):
        optimizer.zero_grad()
        update_lr(optimizer, i, params['learning_rate'])
        x = target_distr(params['batch_size']).to(device)
        x.requires_grad = True

        t = choose_timesteps(params['odeint_method']).to(device)
        l0 = torch.zeros((x.size(0),1), requires_grad=True).to(device)
        initial_values = (x, l0)
        augmented_dynamics = aug_ode(model)

        z_t, log_det = odeint(augmented_dynamics, initial_values, t,
                                        method=params['odeint_method'],        
                                        atol=1e-5,
                                        rtol=1e-5,)
        z1, l1 = z_t[-1], log_det[-1]

        logp_x = initial_distr.log_prob(z1) + l1
        loss = -logp_x.mean()
        loss.backward()
        optimizer.step()

        batch_loss += loss
        if batch_loss > 1e6:
            print(f"\n training collapsed at batch {i}")
            break
        del x, t, z_t, log_det, loss

        if i%1000 == 0:
            batch_loss = batch_loss/1000
            elapsed_time = time.time() - start
            estimated_time = ((elapsed_time)/i)*params['n_batches']
            elapsed_time = format_elapsed_time(elapsed_time)
            progress_bar(i, params['n_batches'], elapsed_time, format_elapsed_time(estimated_time))
            torch.save(model.state_dict(), os.path.join(path + "/models", f"{i}_model.pt"))
            
            data = [[i, batch_loss, elapsed_time]]
            save_logs(path, data, train=True, params=params, first_write=first)
            first = False
            #print(f"\nbatch {i}, Loss: {batch_loss}\n")
            batch_loss = 0

def train_toy_rnode(target, params):

    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)

    target_distr = sampler(target)
    initial_distr = torch.distributions.MultivariateNormal(torch.zeros(2).to(device), torch.eye(2).to(device))
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    model.train()

    batch_loss = 0
    start = time.time()
    first = True
    path = target + "/rnode"
    os.makedirs(path + "/models", exist_ok=True)

    for i in range(1, params['n_batches']+1):
        optimizer.zero_grad()
        
        x = target_distr(params['batch_size']).to(device)
        x.requires_grad = True

        t = choose_timesteps(params['odeint_method']).to(device)
        l0 = torch.zeros((x.size(0),1), requires_grad=True).to(device)
        kin_E0 = torch.zeros((x.size(0),1), requires_grad=True).to(device)
        n0 = torch.zeros((x.size(0),1), requires_grad=True).to(device)
        initial_values = (x, l0, kin_E0, n0)

        augmented_dynamics = reg_aug_toy(model)
        z_t, log_det, E_t, n_t = odeint(augmented_dynamics, initial_values, t,
                                        method=params['odeint_method'],        
                                        atol=1e-5,
                                        rtol=1e-5,)
        z1, l1, kin_E1, n1 = z_t[-1], log_det[-1], E_t[-1], n_t[-1]

        logp_x = initial_distr.log_prob(z1) + l1 - params['lambda_k'] * kin_E1 - params['lambda_j'] * n1
        loss = -logp_x.mean()/2
        loss.backward()
        optimizer.step()

        batch_loss += loss
        del x, t, z_t, log_det, E_t, n_t, loss

        if i%1000 == 0:
            batch_loss = batch_loss/1000
            elapsed_time = time.time() - start
            estimated_time = ((elapsed_time)/i)*params['n_batches']
            elapsed_time = format_elapsed_time(elapsed_time)
            progress_bar(i, params['n_batches'], elapsed_time, format_elapsed_time(estimated_time))
            torch.save(model.state_dict(), os.path.join(path + "/models", f"{i}_model.pt"))
            data = [[i, batch_loss, elapsed_time]]
            save_logs(path, data, train=True, params=params, first_write=first)
            first = False
            #print(f"\nbatch {i}, Loss: {batch_loss}\n")
            batch_loss = 0

def train_toy_cfm(target, params):

    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP().to(device)

    target_distr = sampler(target)
    initial_distr = torch.distributions.MultivariateNormal(torch.zeros(2).to(device), torch.eye(2).to(device))
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    if params["optimal_transport"]:
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=params['sigma'])
        path = target + f"/otcfm{params['sigma']}"
    else:
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=params['sigma'], optimal_transport=False)
        path = target + "/cfm"
    model.train()

    batch_loss = 0
    start = time.time()
    first = True
    os.makedirs(path + "/models", exist_ok=True)

    for i in range(1, params['n_batches']+1):
        optimizer.zero_grad()
        x0, x1 = initial_distr.sample((params['batch_size'],)).to(device), target_distr(params['batch_size']).to(device)
        t, xt, ut = FM.sample_location_and_conditional_flow(x1, x0)
        
        loss = torch.mean((model(t,xt) - ut)**2)
        loss.backward()
        optimizer.step()

        batch_loss += loss
        del x0, x1, t, xt, ut, loss

        if i%1000 == 0:
            batch_loss = batch_loss/1000
            elapsed_time = time.time() - start
            estimated_time = ((elapsed_time)/i)*params['n_batches']
            elapsed_time = format_elapsed_time(elapsed_time)
            progress_bar(i, params['n_batches'], elapsed_time, format_elapsed_time(estimated_time))
            torch.save(model.state_dict(), os.path.join(path + "/models", f"{i}_model.pt"))
            #print(f"\nbatch {i}, Loss: {batch_loss}\n")
            data = [[i, batch_loss, elapsed_time]]
            save_logs(path, data, train=True, params=params, first_write=first)
            first = False
            batch_loss = 0


from plots import generate_grid
def train_mnist_node(params):
    
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if params['activation'] == 'swish':
        activation = Swish()
    elif params['activation'] == 'selu':
        activation = torch.nn.SELU()
    elif params['activation'] == 'softplus':
        activation = torch.nn.Softplus()
    model = UNet(act=activation).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], betas=(0.9, 0.999), weight_decay=0)
    train_loader = mnist_train_loader(params["batch_size"])
    if params['p'] != None:
        path = f"mnist/node{params['p']}"
    else:
        path = f"mnist/node"
    os.makedirs(path + "/models", exist_ok=True)
    start = time.time()
    itr = 0
    first = True

    model.train()

    for epoch in range(1, params['n_epochs']+1):
        epoch_loss = 0.0
        progress_bar = enumerate(train_loader)
        num = 0
        for i, (samples, labels) in progress_bar:
            optimizer.zero_grad()
            if params['learning_rate'] == 1e-3:
                update_lr(optimizer, itr, params['learning_rate'])
            #samples.requires_grad = True

            x0 = samples.to(device)
            
            t = choose_timesteps(params['odeint_method']).to(device)
            l0 = torch.zeros((x0.size(0),1), requires_grad=True).to(device)
            initial_values = (x0, l0)

            augmented_dynamics = aug_ode(model)

            z_t, log_det = odeint(augmented_dynamics, initial_values, t, 
                                            method=params['odeint_method'], 
                                            rtol=1e-5, 
                                            atol=1e-5,)
            
            z1, l1 = z_t[-1], log_det[-1]
            logpz = standard_normal_logprob(z1).view(z1.shape[0], -1).sum(1, keepdim=True).squeeze()
            logpx = logpz + l1
            loss = - torch.sum(logpx) / z1.nelement() 
            #print(z1[0,0])
            #print(standard_normal_logprob(z1)[0,0])
            #print('logrpob and divergence: ', logpz.mean(), l1.mean())
            #loss = compute_bits_per_dim(z1, l1)
            #print(loss)

            loss.backward()
            optimizer.step()
            epoch_loss += loss
            num += 1
            itr += 1

            torch.save(model.state_dict(), os.path.join(path + "/models", f"{epoch}_model.pt"))
            elapsed_time = format_elapsed_time(time.time()-start)
            data = [[epoch, epoch_loss/num, elapsed_time]]
            save_logs(path, data, train=True, params=params, first_write=first)
            first = False
            if num % 50 == 0:
                generate_grid(os.path.join(path + "/models", f"{epoch}_model.pt"))
        generate_grid(os.path.join(path + "/models", f"{epoch}_model.pt"))

    del x0, t, z_t, log_det, loss    

def train_mnist_rnode(params):

    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if params['activation'] == 'swish':
        activation = Swish()
    elif params['activation'] == 'selu':
        activation = torch.nn.SELU()
    elif params['activation'] == 'softplus':
        activation = torch.nn.Softplus()
    model = UNet(act=activation).to(device)

    first = True
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=0)
    train_loader = mnist_train_loader(params["batch_size"])

    if params['p'] != None:
        path = f"mnist/rnode{params['p']}"
    else:
        path = f"mnist/rnode"

    os.makedirs(path + "/models", exist_ok=True)
    start = time.time()
    itr = 0

    model.train()

    for epoch in range(1, params['n_epochs']+1):
        epoch_loss = 0.0
        progress_bar = enumerate(train_loader)
        num = 0
        for i, (samples, labels) in progress_bar:
            optimizer.zero_grad()
            if params['learning_rate'] == 1e-3:
                update_lr(optimizer, itr, params['learning_rate'])
            #samples.requires_grad = True

            x0 = samples.to(device)
            
            t = choose_timesteps(params['odeint_method']).to(device)
            l0 = torch.zeros((x0.size(0),1), requires_grad=True).to(device)
            kin_E0 = torch.zeros((x0.size(0),1), requires_grad=True).to(device)
            n0 = torch.zeros((x0.size(0),1), requires_grad=True).to(device)
            initial_values = (x0, l0, kin_E0, n0)

            augmented_dynamics = reg_aug_mnist(model)

            z_t, log_det, E_t, n_t = odeint(augmented_dynamics, initial_values, t, 
                                            method=params['odeint_method'], 
                                            rtol=1e-5, 
                                            atol=1e-5,)

            z1, l1, kin_E1, n1 = z_t[-1], log_det[-1].squeeze(), E_t[-1].squeeze(), n_t[-1].squeeze()

            #logp_x = compute_bits_per_dim(z1, l1)
            #regularization = (params['lambda_k'] * kin_E1 + params['lambda_j'] * n1).sum() / z1.nelement() 
            #loss = logp_x + regularization

            logpz = standard_normal_logprob(z1).view(z1.shape[0], -1).sum(1, keepdim=True).squeeze()
            loss = (-(logpz + l1) + params['lambda_k'] * kin_E1 + params['lambda_j'] * n1).sum() / z1.nelement() 


            loss.backward()
            optimizer.step()
            epoch_loss += loss
            num += 1
            itr += 1

            elapsed_time = format_elapsed_time(time.time()-start)
            data = [[epoch, epoch_loss/num, elapsed_time]]
            save_logs(path, data, train=True, params=params, first_write=first)
            first = False
            if num % 50 == 0:
                torch.save(model.state_dict(), os.path.join(path + "/models", f"{epoch}_model.pt"))
                generate_grid(os.path.join(path + "/models", f"{epoch}_model.pt"))

        torch.save(model.state_dict(), os.path.join(path + "/models", f"{epoch}_model.pt"))
        generate_grid(os.path.join(path + "/models", f"{epoch}_model.pt"))

        """if epoch%5==0:
            torch.save(model.state_dict(), os.path.join(path + "/models", f"{epoch}_model.pt"))
            elapsed_time = format_elapsed_time(time.time()-start)
            data = [[epoch, epoch_loss/num, elapsed_time]]
            save_logs(path, data, train=True, params=params, first_write=first)
            first = False"""

    del x0, t, z_t, log_det, E_t, n_t, loss    


def train_mnist_cfm(params):

    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    if params["optimal_transport"]:
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=params['sigma'])
        path = "mnist/otcfm"
    else:
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=params['sigma'], optimal_transport=False)
        path = "mnist/cfm"
    train_loader = mnist_train_loader(params["batch_size"])

    os.makedirs(path + "/models", exist_ok=True)
    start = time.time()
    first = True
    model.train()

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
            first = False
        
    del x0, samples, t, xt, ut, loss


def train_model(dataset, training, 
                n_epochs = 1,       
                n_batches = 10,   
                batch_size = 10,
                odeint_method = 'rk4',
                learning_rate = 5e-4,
                lambda_k = .01,       
                lambda_j = .01,       
                sigma = 0.001,
                p = None,
                act = 'swish',
                ot = True):         
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

    params = {
        "seed": 22,
        "learning_rate": learning_rate,
    }    

    if dataset == "mnist":
        params["n_epochs"] = n_epochs
        params["batch_size"] = batch_size
        params["p"] = p
        params["activation"] = act

        if training == "node":
            params["odeint_method"] = odeint_method
            train_mnist_node(params)

        elif training == "rnode":
            params["odeint_method"] = odeint_method
            params["lambda_k"] = lambda_k
            params["lambda_j"] = lambda_j
            train_mnist_rnode(params)

        else:
            params["sigma"] = sigma
            params["optimal_transport"] = ot
            train_mnist_cfm(params)
    else:
        params["n_batches"] = n_batches
        params["batch_size"] = batch_size

        if training == "node":
            params["odeint_method"] = odeint_method
            train_toy_node(dataset, params)

        elif training == "rnode":
            params["odeint_method"] = odeint_method
            params["lambda_k"] = lambda_k
            params["lambda_j"] = lambda_j
            train_toy_rnode(dataset, params)

        else:
            params["sigma"] = sigma
            params["optimal_transport"] = ot
            train_toy_cfm(dataset, params)

#train_model('mnist', 'otcfm')
