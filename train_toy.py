import subprocess

def run_experiment(training_method, dataset):
    cmd = [
        "python", "main.py", 
        "--training-method", training_method,
        "--dataset", dataset,
        "--learning-rate", '1e-3',
        "--n-batches", '100_000',
        "--regularization", '.1',
        "--sigma", '0'
    ]
    
    print(f"Running: {cmd}")

    subprocess.run(cmd)

if __name__ == '__main__':

    training_methods = ['otcfm', 'rnode', 'cfm', 'node']
    datasets = ['moons', 'gaussians', 'spirals', 'checkerboard']

    for training_method in training_methods:
        for dataset in datasets:
            run_experiment(training_method, dataset)
