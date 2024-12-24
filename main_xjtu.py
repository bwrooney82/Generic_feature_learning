import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from model.RSN import mymodel
from offline import train_siamese_network, split_dataset
from online import online_monitoring_statistic, get_embeddings
from utils.augumentations import generate_pairs


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Siamese Network for Anomaly Detection with Online Monitoring")
    parser.add_argument('--data_path', type=str, default='data/data11.npy', help='Path to the dataset')
    parser.add_argument('--train_size', type=int, default=200, help='Number of offline data samples')
    parser.add_argument('--num_pairs', type=int, default=15000, help='Number of pairs to generate for training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--channel', type=int, default=2, help='Channel dimension')
    parser.add_argument('--dim', type=int, default=128, help='Number of epochs to train')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--start', type=int, default=200, help='Starting index for online monitoring')
    parser.add_argument('--history', type=int, default=10, help='History window size for online monitoring')
    parser.add_argument('--current', type=int, default=1, help='Current window size for online monitoring')
    parser.add_argument('--lambda_', type=float, default=0.0010, help='lambda parameter for online monitoring')
    parser.add_argument('--beta_', type=float, default=1.0, help='beta for step size')
    parser.add_argument('--gamma_', type=float, default=1e-2, help='gamma for regularization')
    parser.add_argument('--alpha_', type=float, default=0.10, help='alpha for control limit')
    return parser.parse_args()


def main(mode, model_path):
    # Parse arguments
    args = parse_args()

    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0][4:]
    embedding_dir = os.path.join('results/embedding', dataset_name)
    statistics_dir = os.path.join('results/statistics', dataset_name)
    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(statistics_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    all_data = np.load(args.data_path).transpose((1, 0, 2))
    offline_data = all_data[:args.train_size]
    pairs, labels = generate_pairs(offline_data, num_pairs=args.num_pairs)

    train_pairs, test_pairs, train_targets, test_targets = split_dataset(pairs, labels)

    # Initialize the model
    embedding_net = mymodel(in_channels=args.channel, out_dim=args.dim)
    model = SiameseNet(embedding_net).to(device)
    print(f'Total params: {sum(p.numel() for p in model.parameters()) / 1000000.0:.2f}M')

    if mode == 'train':
        model = train_siamese_network(
            model, train_pairs, train_targets, test_pairs, test_targets,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device, num_workers=args.num_workers
        )
    elif mode == 'test':
        model.load_state_dict(torch.load(model_path))

    embeddings = get_embeddings(model, all_data)
    torch.save(embeddings, os.path.join(embedding_dir, 'Y.pt'))

    R, Cl, tau, Z = online_monitoring_statistic(
        embeddings,
        start=args.start,
        history=args.history,
        current=args.current,
        lambda_=args.lambda_,
        beta=args.beta_,
        gamma=args.gamma_,
        alpha=args.alpha_
    )

    torch.save(Z, os.path.join(embedding_dir, 'Z_lamda{}_beta{}.pt'.format(args.lambda_, args.beta_)))

    np.savez(os.path.join(statistics_dir, 'RHT_lamda{}_beta{}.npz'.format(args.lambda_, args.beta_)), R_list=R,
             Cl_list=Cl, anomalies=tau)


if __name__ == '__main__':
    main(mode='test', model_path='results/save_model/RSN4_XJTU_20epoch.pth')
