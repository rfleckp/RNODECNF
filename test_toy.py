import subprocess

def run_experiment(dataset):
    cmd = [
        "python", "test.py", 
        "--dataset", dataset,
    ]
    print(f"Running: {cmd}")

    subprocess.run(cmd)

if __name__ == '__main__':
    datasets = ['moons', 'gaussians', 'spirals', 'checkerboard']
    for dataset in datasets:
        run_experiment(dataset)
