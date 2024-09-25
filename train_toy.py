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
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print(f"Output:\n{result.stdout}")
        if result.stderr:
            print(f"Errors:\n{result.stderr}")
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {cmd}")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}: {cmd}")
    except Exception as e:
        print(f"An error occurred: {e}")

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
