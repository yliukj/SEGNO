import argparse
import numpy as np
import torch
import os
import random
from torch import nn, optim
import time

from torch_geometric.data import Data
from torch_scatter import scatter
from models.model import SEGNO


from nbody.dataset_nbody import NBodyDataset

time_exp_dic = {'time': 0, 'counter': 0}


def train(device, model, interval, args):


    dataset_train = NBodyDataset(partition='train', dataset_name=args.dataset_dir,
                                 max_samples=args.max_samples)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=0, shuffle=True, drop_last=True)

    dataset_val = NBodyDataset(partition='val', dataset_name=args.dataset_dir)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, num_workers=0, shuffle=False, drop_last=False)

    dataset_test = NBodyDataset(partition='test', dataset_name=args.dataset_dir)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=0, shuffle=False, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_mse = nn.MSELoss()


    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best = {'long_loss': {}}
    for epoch in range(0, args.epochs):
        train_loss, _ = run_epoch(model, optimizer, loss_mse, epoch, loader_train, interval, device, args)
        if (epoch % args.test_interval == 0 or epoch == args.epochs-1) and epoch > 0:
            val_loss, res = run_epoch(model, optimizer, loss_mse, epoch, loader_val, interval, device, args, backprop=False)
            test_loss, res = run_epoch(model, optimizer, loss_mse, epoch, loader_test, interval,
                                  device, args, backprop=False)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_epoch = epoch
                best = res
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d" %
                  (best_val_loss, best_test_loss, best_epoch))


    return best_val_loss, best_test_loss, best_epoch


def run_epoch(model, optimizer, criterion, epoch, loader, interval, device, args, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'counter': 0}
    n_nodes = 5
    batch_size = args.batch_size

    edges = loader.dataset.get_edges(args.batch_size, n_nodes)
    edges = [edges[0], edges[1]]
    edge_index = torch.stack(edges)

    for batch_idx, data in enumerate(loader):
        data = [d.to(device) for d in data]
        for i in range(len(data)):
            if len(data[i].shape) == 4:
                data[i] = data[i].transpose(0, 1).contiguous()
                data[i] = data[i].view(data[i].size(0), -1, data[i].size(-1))
            else:
                data[i] = data[i][:, :data[i].size(1), :].contiguous()
                data[i] = data[i].view(-1, data[i].size(-1))  

        locs, vels, edge_attr, charges, loc_ends = data
        prod_charges = charges[edge_index[0]] * charges[edge_index[1]]

        loc, loc_end, vel = locs[interval[0]], locs[interval[1]], vels[interval[0]]

        if args.time_exp:
            torch.cuda.synchronize()
            t1 = time.time()

        optimizer.zero_grad()

        h = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
        rows, cols = edges
        loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
        edge_attr = torch.cat([prod_charges, loc_dist], 1).detach()  # concatenate all edge properties
        loc_pred, h = model(h, loc.detach(), edges, vel.detach(), edge_attr)
        loss = criterion(loc_pred, loc_end)

        if args.time_exp:
            torch.cuda.synchronize()
            t2 = time.time()
            time_exp_dic['time'] += t2 - t1
            time_exp_dic['counter'] += 1

            if epoch % 100 == 0:
                print("Forward average time: %.6f" % (time_exp_dic['time'] / time_exp_dic['counter']))

        if backprop:    
            loss.backward()
            optimizer.step()

        res['loss'] += loss.item()*batch_size
        res['counter'] += batch_size


    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f' % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter'], res
