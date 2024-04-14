import torch
import argparse
import numpy as np
import random
from models.model import SEGNO


def seed(seed=41):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Run parameters
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-12,
                        help='weight decay')
    parser.add_argument('--print', type=int, default=100,
                        help='print interval')
    parser.add_argument('--log', type=bool, default=False,
                        help='logging flag')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Num workers in dataloader')
    parser.add_argument('--save_dir', type=str, default="saved models",
                        help='Directory in which to save models')

    # Data parameters
    parser.add_argument('--dataset', type=str, default="gravity",
                        help='Data set')
    parser.add_argument('--root', type=str, default="datasets",
                        help='Data set location')
    parser.add_argument('--download', type=bool, default=False,
                        help='Download flag')
    parser.add_argument('--target', type=str, default="medium",
                        help='Target interval[short, medium, long]')
    parser.add_argument('--dataset_dir', type=str, default="dataset_gravity",
                        help='dataset_dir [dataset, dataset_long, dataset_gravity]')
    parser.add_argument('--max_samples', type=int, default=3000,
                        help='Maximum number of samples in nbody dataset')
    parser.add_argument('--time_exp', type=bool, default=False,
                        help='Flag for timing experiment')
    parser.add_argument('--test_interval', type=int, default=5,
                        help='Test every test_interval epochs')

    # Model parameters
    parser.add_argument('--model', type=str, default="segnn",
                        help='Model name')
    parser.add_argument('--hidden_features', type=int, default=64,
                        help='len of hidden rep')
    parser.add_argument('--layers', type=int, default=8,
                        help='Number of message passing layers')
    parser.add_argument('--norm', type=str, default="none",
                        help='Normalisation type [instance, batch]')

    # Parallel computing stuff
    parser.add_argument('-g', '--gpus', default=0, type=int,
                        help='number of gpus to use (assumes all are on one node)')

    args = parser.parse_args()

    if args.gpus == 0:
        device = 'cpu'
        args.mode = 'cpu'
        print('Starting training on the cpu...')
    elif args.gpus == 1:
        args.mode = 'gpu'
        device = torch.device('cuda:0')
        print('Starting training on a single gpu...')

    # Select dataset.
    if args.dataset == "charged":
        from nbody.train_nbody import train

        model = SEGNO(in_node_nf=1, in_edge_nf=2, hidden_nf=64, device=device, n_layers=args.layers,
                      recurrent=True, norm_diff=False, tanh=False)
        if args.target == "short":
            interval = [30, 40]
        elif args.target == "medium":
            interval = [30, 45]
        elif args.target == "long":
            interval = [30, 50]

        train(device, model, interval, args)
    elif args.dataset == "gravity":
        from nbody.train_gravity import train

        model = SEGNO(in_node_nf=1, in_edge_nf=1, hidden_nf=64, device=device, n_layers=args.layers,
                      recurrent=True, norm_diff=False, tanh=False)
        if args.target == "short":
            interval = [0, 10]
        elif args.target == "medium":
            interval = [0, 15]
        elif args.target == "long":
            interval = [0, 20]
        train(device, model, interval, args)
    else:
        raise Exception("Dataset could not be found")
