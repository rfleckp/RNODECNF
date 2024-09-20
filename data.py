import torch
from torchdyn.datasets import generate_gaussians, generate_moons, generate_spirals
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import numpy as np


#Toy data sampler

#Toy data sampler

def sample_mix_normal(num_samples, means=[torch.tensor([20, 5.0]), torch.tensor([20, -5.0])], covariances=[torch.eye(2), torch.eye(2)]):
    assert len(means) == len(covariances)
    num_components = len(means)
    component_choices = torch.randint(0, num_components, (num_samples,))

    samples = []
    for i in component_choices:
        mean = means[i]
        covariance = covariances[i]

        distribution = torch.distributions.MultivariateNormal(mean, covariance)
        sample = distribution.sample()
        samples.append(sample)
    
    return torch.stack(samples)

def sample_gaussians(n, n_gaussians=8, radius=4):
    num = int(n/n_gaussians)
    x0, _ = generate_gaussians(n_samples=num, n_gaussians=n_gaussians, radius=radius)
    del _
    return x0

def sample_moons(n):
    x0, _ = generate_moons(n, noise=0.2)
    del _
    return x0 * 3 - 1

def sample_spirals(n):
    x0, _ = generate_spirals(int(n/2))
    del _
    return x0

def sample_checkerboard(n):
    centers = [(-1.5, 1.5), (0.5, 1.5), (-0.5, 0.5), (1.5, 0.5), 
                    (-1.5, -0.5), (0.5, -0.5), (-0.5, -1.5), (1.5, -1.5)]
    dataset = []
    for i in range(n):
        point = np.random.uniform(-.5, .5, size=2)
        center = random.choice(centers)
        point += center
        dataset.append(point)
    dataset = torch.tensor(np.array(dataset, dtype='float32') * 2)

    return dataset

class sampler():
    def __init__(self, dataset: str):
        if dataset == "normal":
            self.sampler = sample_mix_normal
        elif dataset == "gaussians":
            self.sampler = sample_gaussians
        elif dataset == "moons":
            self.sampler = sample_moons
        elif dataset == "spirals":
            self.sampler = sample_spirals
        elif dataset == "checkerboard":
            self.sampler = sample_checkerboard
        else:
            raise Exception("Selected Dataset not supported. Choose between normal, gaussians, moons, checkerboard or spirals")

    
    def __call__(self, n: int):
        return self.sampler(n)


def mnist_train_loader(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def mnist_test_loader(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)