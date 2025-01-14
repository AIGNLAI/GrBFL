from datamanger import process_data, load_or_process_data,generate_ratio
from config import get_args
from train import fed_train
from nets.GCNnet import *
from torch_geometric.loader import DataLoader
import numpy as np
import os
import random

if __name__ == '__main__':
    args = get_args()

    # Define dataset paths
    train_dir = f'./data/{args.dataset}/train/'
    test_dir = f'./data/{args.dataset}/test/'
    train_file = f'train_data_{args.dataset}_{args.num_clients}.dill'
    test_file = f'test_data_{args.dataset}_{args.num_clients}.dill'

     # Step 1: Count files in the training directory
    with os.scandir(train_dir) as entries:
        file_count = sum(entry.is_file() for entry in entries)
    assert file_count > 0, "Training directory is empty or invalid!"

    samples_per_client = file_count // args.num_clients
    remainder = file_count % args.num_clients

    samplenum = [samples_per_client] * args.num_clients
    for i in range(remainder):
        samplenum[i] += 1

    presam = [0]
    for i in range(args.num_clients):
        presam.append(presam[-1] + samplenum[i])
    presam.pop(0)

    div = list(range(file_count))
    random.shuffle(div)

    mp = {}
    idx = 0
    for i in div:
        for j in range(args.num_clients):
            if idx < presam[j]:
                mp[i] = j
                break
        idx += 1
    assert all(i in mp for i in range(file_count)), "Mapping `mp` generation failed!"


    # Step 3: Process data
    train_list = load_or_process_data(train_file, process_data, train_dir, args.num_clients, mp)
    test_list = load_or_process_data(test_file, process_data, test_dir, args.num_clients, mp, is_train=False)

    train_loader = [DataLoader(data, batch_size=args.batchSize, shuffle=True) for data in train_list]
    test_loader = DataLoader(test_list, batch_size=args.batchSize)

    # Step 4: Start federated training
    fed_train(num_client=args.num_clients, dataset=args.dataset, data_loader_fn=train_loader,
              model_fn=lambda: GCN_8_plus(num_features=8, num_classes=10), 
              train_list=train_list, test_loader=test_loader, args=args, samplenum = samplenum)

   