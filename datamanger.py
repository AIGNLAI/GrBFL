import os
import numpy as np
import h5py
from tqdm import tqdm
import torch
from torch_geometric.data import Data
import dill

def process_data(data_dir, num_client, mp, is_train=True):
    data_list = [[] for _ in range(num_client)] if is_train else []
    idx = 0
    for file_name in tqdm(os.listdir(data_dir), desc=f"Processing {'training' if is_train else 'test'} data"):
        graph_dir = os.path.join(data_dir, file_name)
        if h5py.is_hdf5(graph_dir):
            with h5py.File(graph_dir, 'r') as f:
                all = np.array(f['x'])
                a2 = all[:, :5]
                b = all[:, 6:9]
                Center2 = np.hstack((a2, b))
                data = Data(x=torch.tensor(Center2, dtype=torch.float),
                            edge_index=torch.tensor(np.array(f['edge_index']), dtype=torch.long),
                            y=torch.tensor(np.array(f['y']), dtype=torch.long))
                if is_train:
                    data_list[mp[idx]].append(data)
                else:
                    data_list.append(data)
            idx += 1
    return data_list

def load_or_process_data(file_path, process_fn, *args, **kwargs):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return dill.load(f)
    else:
        data = process_fn(*args, **kwargs)
        with open(file_path, 'wb') as f:
            dill.dump(data, f)
        return data

def generate_ratio(num_clients, min_ratio=0.1):
    assert 0 < min_ratio < 1 / num_clients, "min_ratio must be smaller than 1/num_clients."
    while True:
        ratio = np.random.dirichlet(np.ones(num_clients) * 10)
        if (ratio >= min_ratio).all():
            return ratio
