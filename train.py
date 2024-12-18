import torch
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from torch import nn
import torch.optim as optim

def test_model(device, net, test_loader, criterion):
    net.eval()
    probs_score = []
    labels_truth = []
    loss_sum = 0
    num_correct = 0
    for data in test_loader:
        labels = data.y

        # Ensure data and labels are on the correct device
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
    
    eval_loss = loss_sum / len(test_loader)
    eval_acc = num_correct / len(test_loader.dataset)
    
    return eval_loss, eval_acc

def fed_train(num_client, dataset, data_loader_fn, model_fn, train_list, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = model_fn()
    global_model.to(device)
    criterion = nn.CrossEntropyLoss()
    global_params = global_model.state_dict()

    for epoch in range(args.epochs):
        # Train on each client
        local_models = []
        for client_idx in range(num_client):
            client_model = model_fn()
            client_model.to(device)
            client_model.load_state_dict(global_params)
            optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)

            for data in data_loader_fn[client_idx]:
                client_model.train()
                if torch.cuda.is_available():
                    data = data.to(device)  # Ensure data is moved to the correct device
                optimizer.zero_grad()
                output = client_model(data)
                loss = criterion(output, data.y)
                loss.backward()
                optimizer.step()
            local_models.append(client_model.state_dict())

        # Aggregate updates
        global_params = aggregate_parameters(local_models)
        global_model.load_state_dict(global_params)

        # Evaluate global model
        eval_loss, eval_acc = test_model(device, global_model, test_loader, criterion)
        print(f"Epoch {epoch+1}: Loss = {eval_loss}, Accuracy = {eval_acc}")

def aggregate_parameters(local_models):
    global_params = local_models[0]
    for key in global_params.keys():
        global_params[key] = global_params[key].float()
        for model_state in local_models[1:]:
            global_params[key] += model_state[key].float()
        global_params[key] /= len(local_models)
    return global_params

