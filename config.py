import argparse

def get_args():
    parser = argparse.ArgumentParser('Configuration for Federated Learning')
    parser.add_argument('--batchSize', default=96, type=int, help='Batch size for training')
    parser.add_argument('--dataset', default="CIFAR10", type=str, help='Dataset to use')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs for training')
    parser.add_argument('--num_clients', default=3, type=int, help='Number of clients in federated learning')  # 添加 num_clients 参数
    parser.add_argument('--model', default='10new_Net3_maxmin_10step_bS192', type=str, help='Model name')
    parser.add_argument('--checkpoint_dir', default='checkpoints', type=str, help='Output directory for checkpoints')
    parser.add_argument('--need_step', default=100000, type=int, help='Number of steps required for training')
    return parser.parse_args()
