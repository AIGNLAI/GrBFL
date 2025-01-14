import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def fed_train(num_client, dataset, data_loader_fn, model_fn, train_list, test_loader, args,samplenum):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = model_fn()
    global_model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    reg_lambda = 1e-3 

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        global_params = global_model.state_dict()
        local_models = []
        for client_idx in range(num_client):
            client_model = model_fn()
            client_model.load_state_dict(global_params)
            client_model.to(device)
            optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
            with tqdm(total=10, desc=f"Training Client {client_idx + 1}", unit="epoch") as pbar:
                for local_epoch in range(10):
                    epoch_loss = 0.0
                    epoch_samples = 0
                    
                    for data in data_loader_fn[client_idx]:
                        client_model.train()
                        if torch.cuda.is_available():
                            data = data.to(device)
                        
                        output = client_model(data).squeeze()
                        loss = criterion(output, data.y)
                        
                        reg_loss = 0
                        client_dict = client_model.state_dict()
                        for k in client_dict.keys():
                            reg_loss += torch.sum(abs(client_dict[k] - global_params[k]))
                        loss = loss + reg_lambda * reg_loss
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        epoch_samples += len(data.y)

                    avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
                    pbar.set_postfix({"Avg Loss": avg_loss, "LR": optimizer.param_groups[0]['lr']})
                    pbar.update(1)
                eval_loss, eval_acc = test_model(device, client_model, test_loader, criterion)
                print(f"\nTraining Loss = {eval_loss:.4f}, Traning Accuracy = {eval_acc:.4f}")
                local_models.append(client_model.state_dict())

        averaged_params = local_models[0]
        for k in averaged_params.keys():
            for i in range(num_client):
                local_sample_number = samplenum[i]
                local_model_params = local_models[i]
                w = local_sample_number / sum(samplenum)
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        global_model.load_state_dict(averaged_params)

        eval_loss, eval_acc = test_model(device, global_model, test_loader, criterion)
        print(f"Epoch {epoch + 1}: Loss = {eval_loss:.4f}, Accuracy = {eval_acc:.4f}")

    print("Training completed.")


def test_model(device, model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data in test_loader:
            if torch.cuda.is_available():
                data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y)
            total_loss += loss.item() * len(data.y)
            preds = output.argmax(dim=1)
            total_correct += (preds == data.y).sum().item()
            total_samples += len(data.y)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy
