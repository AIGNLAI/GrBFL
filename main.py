# -*- coding: utf-8 -*-
# @Time    : 2022/9/7 16:45
# @Author  : lan
# @Software: PyCharm
import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt
from torch import nn

from nets.Net5 import GIN
from nets.Net4 import GCN,GCNNet,Gat
from nets.Net3 import GAT_3
from nets.Net2 import GAT_2
from nets.Net1 import GAT_1
from collections import Counter

''' For Keras dataset_load()'''
import torch

from tqdm import tqdm

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import matplotlib


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
            loss_sum += loss.item()  # * label.size(0)

            # 存储预测值的分数和目标标签值# 保存以绘制auc曲线
            preds10_score = torch.softmax(outputs, dim=1)
            targets = labels.detach().cpu().numpy()
            preds2 = np.argmax(preds10_score.detach().cpu().numpy(), axis=1)

            probs_score.append(preds2)
            labels_truth.append(targets)

            corre = (preds2 == targets).sum()

            num_correct += corre.item()


    eval_loss = loss_sum / (len(test_loader))  # num / bS
    eval_acc = num_correct / 10000 # num


    probs_score = np.hstack(probs_score)
    print(Counter(probs_score))
    labels_truth = np.hstack(labels_truth)

    return eval_loss, eval_acc, probs_score, labels_truth

import os

import torch


def save_checkpoint(best_acc, model, optimizer, step):
    print('Best Model Saving...')
    model_state_dict = model.state_dict()
    if not os.path.isdir(args.checkpoint_dir + '/' + args.model):
        os.mkdir(args.checkpoint_dir + '/' + args.model)
    torch.save({
        'model_state_dict': model_state_dict,  # 网络参数
        'global_epoch': step,  # 对应epoch
        'optimizer_state_dict': optimizer.state_dict(),  # 优化器参数
        'best_acc': best_acc,
    }, os.path.join(args.checkpoint_dir + '/' + args.model, 'checkpoint_{}_best_{}.pth'.format(step, best_acc)))


from sklearn.metrics import roc_curve, roc_auc_score, auc
import torch.optim as optim
import argparse


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


import h5py
from tensorboardX import SummaryWriter


def train(dataset = "MNIST"):
    
    best_acc = 0.0  # 记录测试最高准确率
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 指定CPU or cuda
    # device = torch.device("cuda:4")
    #print(device)
    net = GAT_3(num_features=8, num_classes=10)
    net.train()
    #print(net)
    num_params = sum(p.numel() for p in net.parameters())
    print("== Total number of parameters: {}".format(num_params))
    num_params_update = sum(
        p.numel() for p in net.parameters() if p.requires_grad)
    print("== Total number of learning parameters: {}".format(num_params_update))
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.2, momentum=0.9, nesterov=True)
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    test_acc = 0  # 初始的比较值
    trainlist = []
    testlist = []
    if dataset == "MNIST":
        train_dir1 = './data/MNIST/h5_attr_all/train/'
        test_dir = "./data/MNIST/h5_attr_all/test/"
    elif dataset == "CIFAR10":
        train_dir1 = './data/CIFAR10/h5_attr_all/train/'
        test_dir = './data/CIFAR10/h5_attr_all/test/'
    for train_name1 in os.listdir(train_dir1):
        graph_dir = os.path.join(train_dir1, train_name1)
        f = h5py.File(graph_dir, 'r')
        all = np.array(f['x'])
        a2 = all[:, :5]
        b = all[:, 6:9]  # 7max,8min
        # c = b.reshape((len(a2),1))
        Center2 = np.hstack((a2, b))
        trainlist.append(
            Data(x=torch.tensor(Center2, dtype=torch.float),
                edge_index=torch.tensor(np.array(f['edge_index']), dtype=torch.long),
                edge_attr = torch.tensor(np.array(f['edge_attr']),dtype = torch.long),
                y=torch.tensor(np.array(f['y']), dtype=torch.long))
            )
        f.close()
    for test_name in os.listdir(test_dir):
        graph_dir = os.path.join(test_dir, test_name)
        f = h5py.File(graph_dir, 'r')
        all =np.array(f['x'])
        a2 = all[:, :5]
        b = all[:, 6:9]  # 7max,8min
        Center2 = np.hstack((a2, b))
        testlist.append(
            Data(x=torch.tensor(Center2, dtype=torch.float),
                edge_index=torch.tensor(np.array(f['edge_index']), dtype=torch.long),
                edge_attr = torch.tensor(np.array(f['edge_attr']),dtype = torch.long),
                y=torch.tensor(np.array(f['y']), dtype=torch.long))
            )
        f.close()
    train_loader = DataLoader(trainlist, batch_size=args.batchSize, shuffle=True)  # ,drop_last=True
    test_loader = DataLoader(testlist, batch_size=args.batchSize)
    train_Loss = []
    eval_Loss = []
    train_Acc = []
    eval_Acc = []
    x = []
    train_writer_loss = SummaryWriter('./runs/'+args.model+'/tl')
    train_writer_acc = SummaryWriter('./runs/'+args.model+'/ta')
    eval_writer_loss = SummaryWriter('./runs/'+args.model+'/el')
    eval_writer_acc = SummaryWriter('./runs/'+args.model+'/ea')
    lr = SummaryWriter('./runs/'+args.model+'lr')
    total_step = 0
    acc_before = 0
    n_batches = len(train_loader)
    acc = 0
    train_loss = 0
    num_correct = 0
    count_k = 0
    max_k = 0
    while total_step < args.need_step:
        for step, data in enumerate(train_loader):  # 一个epoch,bS是128，那就60000/128=468.75，总共有56个batch一个epoch的话
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
            if total_step % 50 == 0:
                print('step %.f,Train loss: %.4f, Train acc: %.4f, epoch_cor: %d,' % (total_step,loss.item(),epoch_cor / args.batchSize,epoch_cor,))
            if total_step % 10 == 0:
                eval_loss, eval_acc, probs_score, labels_truth = tes_t(device, net, test_loader, criterion)
                if max_k > eval_acc and count_k < 500:
                    count_k += 1
                if max_k <= eval_acc and count_k < 500:
                    count_k = 0
                    max_k = eval_acc
                if count_k == 500 and optimizer.param_groups[0]['lr'] > 0.01:
                    optimizer.param_groups[0]['lr'] /= 4
                if acc_before <= eval_acc:
                    acc_before = eval_acc
                    save_checkpoint(acc_before, net, optimizer, total_step)
                print('step %.f,Test loss: %.4f, Test acc: %.4f' % (total_step, eval_loss, eval_acc))
                train_writer_loss.add_scalar("loss_train", loss.item(), total_step)
                eval_writer_loss.add_scalar("loss_eval", eval_loss, total_step)
                train_writer_acc.add_scalar("acc_train", epoch_cor / args.batchSize, total_step)
                eval_writer_acc.add_scalar("acc_eval", eval_acc, total_step)
                lr.add_scalar("lr", optimizer.param_groups[0]['lr'], total_step)
                train_writer_loss.close()
                eval_writer_loss.close()
                train_writer_acc.close()
                eval_writer_acc.close()
                lr.close()
                train_Loss.append(loss.item())
                eval_Acc.append(eval_acc)
                x.append(total_step)
                train_Acc.append(epoch_cor / args.batchSize)
                eval_Loss.append(eval_loss)
            total_step += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print(len(train_Loss))
    print(len(eval_Acc))
    print(len(x))
    # 绘制损失和准确度曲线
    plt.title('Loss and Accuracy')
    plt.xlabel('epoch')
    plt.plot(x, train_Loss, 'yellow')
    plt.plot(x, train_Acc, 'cyan')
    plt.plot(x, eval_Loss, 'red')
    plt.plot(x, eval_Acc, 'blue')
    plt.legend(['train_Loss', 'train_Acc', 'eval_Loss', 'eval_Acc'])
    plt.savefig(args.model+".png")
    print("Best ACC:", acc_before)

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
    ##Data
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
        b = all[:, 6:9]  # 7max,8min
        # c = b.reshape((len(a2),1))
        Center2 = np.hstack((a2, b))
        trainlist[mp[idx]].append(
            Data(x=torch.tensor(Center2, dtype=torch.float),
                edge_index=torch.tensor(np.array(f['edge_index']), dtype=torch.long),
                #edge_attr = torch.tensor(np.array(f['edge_attr']),dtype = torch.long),
                y=torch.tensor(np.array(f['y']), dtype=torch.long))
            )
        idx += 1
        if idx%10000 == 0:
            print(idx,"ok!")
        f.close()
        
    for test_name in os.listdir(test_dir):
        graph_dir = os.path.join(test_dir, test_name)
        f = h5py.File(graph_dir, 'r')
        all =np.array(f['x'])
        a2 = all[:, :5]
        b = all[:, 6:9]  # 7max,8min
        Center2 = np.hstack((a2, b))
        testlist.append(
            Data(x=torch.tensor(Center2, dtype=torch.float),
                edge_index=torch.tensor(np.array(f['edge_index']), dtype=torch.long),
                #edge_attr = torch.tensor(np.array(f['edge_attr']),dtype = torch.long),
                y=torch.tensor(np.array(f['y']), dtype=torch.long))
            )
        f.close()
    train_loader = [DataLoader(trainlist[i], batch_size=args.batchSize, shuffle=True) for i in range(num_client)]
    test_loader = DataLoader(testlist, batch_size=args.batchSize)
    cilent = [GAT_3(num_features=8, num_classes=10) for _ in range(num_client)]
    std_parm = cilent[0].state_dict()
    for k in std_parm.keys():
        std_parm[k] = random.uniform(0,1)
    lamda = 1e-6
    #cilent = [GIN() for _ in range(num_client)]
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
                for step, data in enumerate(train_loader[i]):  # 一个epoch,bS是128，那就60000/128=468.75，总共有56个batch一个epoch的话
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
                    #if total_step % 50 == 0:
                        #print('step %.f,Train loss: %.4f, Train acc: %.4f, epoch_cor: %d,' % (total_step,loss.item(),epoch_cor / args.batchSize,epoch_cor,))
                    model_loss = 0
                    state_dict = net.state_dict()
                    for k in std_parm.keys():
                        model_loss += torch.sum(abs(std_parm[k] - state_dict[k]))
                    #print(loss)
                    loss = loss + model_loss*lamda
                    
                    train_Loss.append(loss.item())
                    x.append(total_step)
                    train_Acc.append(epoch_cor / args.batchSize)
                    total_step += 1
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    for p in net.parameters():
                        p.data.clamp_(0, 1)
                #print(loss)
            eval_loss, eval_acc, probs_score, labels_truth = tes_t(device, net, test_loader,criterion)
            print("cilent:",i,"acc:",eval_acc)
            plt.title('Loss and Accuracy')
            plt.xlabel('epoch')
            plt.plot(x, train_Loss, 'yellow')
            plt.plot(x, train_Acc, 'cyan')
            plt.legend(['train_Loss', 'train_Acc'])
            plt.savefig(args.model+".png")
            # 获取模型的参数字典
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
        
        #global_model = GIN()
        global_model = GAT_3(num_features=8, num_classes=10)
        global_model.load_state_dict(averaged_params)
        global_model.to(device)
        eval_loss, eval_acc, probs_score, labels_truth = tes_t(device, global_model, test_loader,criterion)
        print("round:",epoch,eval_loss, eval_acc,probs_score,labels_truth)
    

if __name__ == '__main__':
    args = get_args()
    fedtrain(num_client = 5,dataset="MNIST")
   




