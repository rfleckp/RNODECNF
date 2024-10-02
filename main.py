import os
import argparse
from training import train_model
from test import evaluate_models, evaluate_dataset


def main():
    """
    dataset: ['mnist', 'moons', 'gaussians', 'spirals']
    training method: ['cfm', 'rnode', 'node']
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-method", default="cfm",
                        choices=["rnode", "otcfm", "cfm", "node"])
    
    parser.add_argument("--dataset", default="normal", 
        choices=["moons", "gaussians", "spirals", "mnist", "checkerboard", "normal"])
    
    parser.add_argument("--n-batches", default = 10000, type=int)  
    parser.add_argument("--batch-size", default = 256, type=int)  
    parser.add_argument("--n-epochs", default = 10, type=int)
    parser.add_argument("--learning-rate", default = 5e-4, type=float)  
    parser.add_argument("--odeint-method", default = 'rk4', type=str)    

    parser.add_argument("--sigma", default = 0.001, type=float)
    parser.add_argument("--regularization", default = 0.01, type=float)       

    parser.add_argument("--only-test", default = False, type=bool)
    parser.add_argument("--path", default = None, type=str)
    parser.add_argument("--activation", default = 'swish', type=str,
                        choices=["swish", "selu", "softplus"])
    args = parser.parse_args()
    #print('args parsed')

    ot = False
    if args.training_method=='otcfm':
        #training_method = f'otcfm{args.sigma}'
        ot = True
        
    if not args.only_test:
        print(f"\n\nTRAINING {args.training_method} for {args.dataset}")
 
        train_model(args.dataset, args.training_method, 
                    n_batches=args.n_batches, 
                    n_epochs=args.n_epochs,
                    batch_size=args.batch_size, 
                    odeint_method=args.odeint_method,
                    learning_rate=args.learning_rate,
                    lambda_k=args.regularization,
                    lambda_j= args.regularization,
                    sigma=args.sigma,
                    p=args.path,
                    act=args.activation,
                    ot=ot)

        evaluate_models(os.path.join(args.dataset, args.training_method, 'models'))
    
    else:
        print(f"\n\nTEST {args.dataset}")
        evaluate_dataset(args.dataset)

if __name__ == "__main__":
    main()