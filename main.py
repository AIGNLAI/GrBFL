import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt
from torch import nn
from nets.GATnet import GAT_small,GAT_big
from nets.GCNnet import GCN_small,GCN_big
from nets.GINnet import GIN_small,GIN_big
from nets.GraphSage import GraphSage_small,GraphSage_big
import os
import torch
from collections import Counter
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

def tes_t(device, net, test_loader, criterion):
    net.eval()
    probs_score = []
    labels_truth = []
    Sig = nn.Sigmoid()
    loss_sum = 0
    num_correct = 0
    for data in test_loader:
        labels = data.y
        if torch.cuda.is_available():
            data = data.to(device)
            labels = labels.to(device)
        with torch.no_grad():
            outputs = net(data).squeeze()
            loss = criterion(outputs, labels)
            loss_sum += loss.item() 
            preds10_score = torch.softmax(outputs, dim=1)
            targets = labels.detach().cpu().numpy()
            preds2 = np.argmax(preds10_score.detach().cpu().numpy(), axis=1)
            probs_score.append(preds2)
            labels_truth.append(targets)
            corre = (preds2 == targets).sum()
            num_correct += corre.item()
    eval_loss = loss_sum / (len(test_loader))
    eval_acc = num_correct / 10000 
    probs_score = np.hstack(probs_score)
    print(Counter(probs_score))
    labels_truth = np.hstack(labels_truth)
    return eval_loss, eval_acc, probs_score, labels_truth

import torch.optim as optim
import argparse
import h5py
import random
from copy import deepcopy
def fedtrain(num_client = 3,dataset = "MNIST"):
    args = get_args()
    global_epoch = 1
    best_acc = 0.0  # 记录测试最高准确率
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 指定CPU or cuda
    print(device)
    criterion = nn.CrossEntropyLoss()
    if dataset == "MNIST":
        train_dir1 = './data/MNIST/h5_attr_all/train/'
        test_dir = "./data/MNIST/h5_attr_all/test/"
    elif dataset == "CIFAR10":
        train_dir1 = './data/CIFAR10/h5_attr_all/train/'
        test_dir = './data/CIFAR10/h5_attr_all/test/'
    else:
        train_dir1 = './data/CIFAR100/h5_attr_all/train/'
        test_dir = './data/CIFAR100/h5_attr_all/test/'
    with os.scandir(train_dir1) as entries:
        file_count = sum(entry.is_file() for entry in entries)
    ratio = np.random.dirichlet(np.ones(num_client), 1)
    if num_client < 8:
        while (ratio < 1/num_client*0.8).any():
            ratio = np.random.dirichlet(np.ones(num_client), 1)
    samplenum = []
    for i in range(num_client - 1):
        samplenum.append(int(ratio[0][i]*file_count))
    samplenum.append(file_count - sum(samplenum))
    presam = [0]
    for i in range(num_client):
        presam.append(presam[-1] + samplenum[i])
    presam.pop(0)
    div = [i for i in range(file_count)]
    random.shuffle(div)
    mp = {}
    idx = 0
    for i in div:
        for j in range(num_client):
            if idx < presam[j]:
                break
        mp[i] = j
        idx += 1
    print("比率为：",ratio)
    trainlist = [[] for _ in range(num_client)]
    testlist = []
    idx = 0
    for train_name1 in os.listdir(train_dir1):
        graph_dir = os.path.join(train_dir1, train_name1)        
        f = h5py.File(graph_dir, 'r')
        all = np.array(f['x'])
        a2 = all[:, :5]
        b = all[:, 6:9]
        Center2 = np.hstack((a2, b))
        trainlist[mp[idx]].append(
            Data(x=torch.tensor(Center2, dtype=torch.float),
                edge_index=torch.tensor(np.array(f['edge_index']), dtype=torch.long),
                y=torch.tensor(np.array(f['y']), dtype=torch.long)))
        idx += 1
        if idx%10000 == 0:
            print(idx,"ok!")
        f.close()
    for test_name in os.listdir(test_dir):
        graph_dir = os.path.join(test_dir, test_name)
        f = h5py.File(graph_dir, 'r')
        all =np.array(f['x'])
        a2 = all[:, :5]
        b = all[:, 6:9]
        Center2 = np.hstack((a2, b))
        testlist.append(
            Data(x=torch.tensor(Center2, dtype=torch.float),
                edge_index=torch.tensor(np.array(f['edge_index']), dtype=torch.long),
                y=torch.tensor(np.array(f['y']), dtype=torch.long)))
        f.close()
    train_loader = [DataLoader(trainlist[i], batch_size=args.batchSize, shuffle=True) for i in range(num_client)]
    test_loader = DataLoader(testlist, batch_size=args.batchSize)
    cilent = [GAT_small(num_features=8, num_classes=10) for _ in range(num_client)]
    std_parm = cilent[0].state_dict()
    for k in std_parm.keys():
        std_parm[k] = random.uniform(0,1)
    lamda = 1e-6
    for epoch in range(global_epoch):
        for i in range(num_client):
            net = cilent[i]
            net.train()
            net.to(device)
            optimizer = optim.SGD(net.parameters(), lr=0.2, momentum=0.9, nesterov=True)
            criterion = nn.CrossEntropyLoss()
            train_Loss = []
            train_Acc = []
            total_step = 0
            num_correct = 0
            x = []
            print("train:cilent",i)
            while total_step < args.need_step:
                for step, data in enumerate(train_loader[i]):
                    net.train()
                    label = data.y
                    if torch.cuda.is_available():
                        data = data.to(device)
                        label = label.to(device)
                    out = net(data).squeeze()
                    loss = criterion(out, label)
                    pred_label = np.argmax(out.detach().cpu().numpy(), axis=1)
                    target_l = label.detach().cpu().numpy()
                    epoch_cor = (pred_label == target_l).sum()
                    num_correct += epoch_cor
                    model_loss = 0
                    state_dict = net.state_dict()
                    for k in std_parm.keys():
                        model_loss += torch.sum(abs(std_parm[k] - state_dict[k]))
                    loss = loss + model_loss*lamda
                    train_Loss.append(loss.item())
                    x.append(total_step)
                    train_Acc.append(epoch_cor / args.batchSize)
                    total_step += 1
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            eval_loss, eval_acc, probs_score, labels_truth = tes_t(device, net, test_loader,criterion)
            print("cilent:",i,"acc:",eval_acc)
            plt.title('Loss and Accuracy')
            plt.xlabel('epoch')
            plt.plot(x, train_Loss, 'yellow')
            plt.plot(x, train_Acc, 'cyan')
            plt.legend(['train_Loss', 'train_Acc'])
            plt.savefig(args.model+".png")
            with open("./model_parameters" + str(i)+ ".txt", "w", encoding="utf-8") as f:
                f.write(str(state_dict))
        averaged_params = cilent[0].state_dict()
        
        for k in averaged_params.keys():
            for i in range(num_client):
                local_sample_number = samplenum[i]
                local_model_params = cilent[i].state_dict()
                w = local_sample_number / sum(samplenum)
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w 
                else:
                    averaged_params[k] += local_model_params[k] * w
        global_model = GAT_small(num_features=8, num_classes=10)
        global_model.load_state_dict(averaged_params)
        global_model.to(device)
        eval_loss, eval_acc, probs_score, labels_truth = tes_t(device, global_model, test_loader,criterion)
        print("round:",epoch,eval_loss, eval_acc,probs_score,labels_truth)

def get_args():
    parser = argparse.ArgumentParser('data_pre')
    parser.add_argument('--batchSize', default=96, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--model', default='10new_Net3_maxmin_10step_bS192', type=str, help='model name')
    parser.add_argument('--checkpoint_dir', default='checkpoints', type=str, help='Output directory')
    parser.add_argument('--need_step', default=10000, type=int,
                        help='the all of step(like the bS=60000/step per epoch, and bS is 128,the step is 468.75 per epoch),if epoch is 150 then 468.75*150=70312 step')

    arguments = parser.parse_args()
    return arguments

if __name__ == '__main__':
    args = get_args()
    fedtrain(num_client = 5,dataset="MNIST")
   




