import subprocess

def run_experiment(training_method, dataset, sigma):
    cmd = [
        "python", "main.py", 
        "--training-method", training_method,
        "--dataset", dataset,
        "--learning-rate", '1e-3',
        "--n-batches", '100_000',
        "--regularization", '.1',
        "--sigma", f'{sigma}'
    ]
    
    print(f"Running: {cmd}")

    subprocess.run(cmd)

if __name__ == '__main__':

    training_methods = ['otcfm', 'rnode', 'cfm', 'node']
    sigmas = [1e-1, 1e-3, 1e-5]
    datasets = ['moons', 'gaussians', 'spirals', 'checkerboard']

    '''for training_method in training_methods:
        for dataset in datasets:
            run_experiment(training_method, dataset)'''
    for dataset in datasets:
        for sigma in sigmas:
            run_experiment('otcfm', dataset, sigma)
