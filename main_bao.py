import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from model.RSN import mymodel
from offline import train_siamese_network, split_dataset
from online import online_monitoring_statistic, get_embeddings
from utils.augumentations import generate_pairs
import matplotlib.pyplot as plt
import pywt

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
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--channel', type=int, default=2, help='Channel dimension')
    parser.add_argument('--dim', type=int, default=128, help='Number of epochs to train')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--start', type=int, default=200, help='Starting index for online monitoring')
    parser.add_argument('--history', type=int, default=10, help='History window size for online monitoring')
    parser.add_argument('--current', type=int, default=1, help='Current window size for online monitoring')
    parser.add_argument('--lambda_', type=float, default=0.5, help='lambda parameter for online monitoring')
    parser.add_argument('--beta_', type=float, default=0.1, help='beta for step size')
    parser.add_argument('--gamma_', type=float, default=1e-2, help='gamma for regularization')
    parser.add_argument('--alpha_', type=float, default=0.10, help='alpha for control limit')
    return parser.parse_args()


def denoise_signal(signal, wavelet='db4', level=3):
    """
    使用小波变换对信号进行去噪
    Args:
        signal: 输入信号 (2048,)
        wavelet: 小波基函数类型
        level: 分解层数
    """
    # 小波分解
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # 计算阈值
    threshold = np.sqrt(2 * np.log(len(signal))) * np.median(np.abs(coeffs[-1])) / 0.6745
    
    # 对系数进行软阈值处理
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
    
    # 小波重构
    denoised_signal = pywt.waverec(coeffs, wavelet)
    
    return denoised_signal[:len(signal)]  # 确保输出长度与输入相同


def plot_denoise_comparison(original, denoised, sample_idx=0, channel_idx=0):
    """可视化对比原始信号和去噪后的信号"""
    plt.figure(figsize=(12, 6))
    plt.plot(original[sample_idx, channel_idx], label='Original', alpha=0.7,linewidth=0.1)
    plt.plot(denoised[sample_idx, channel_idx], label='Denoised', alpha=0.7,linewidth=0.5)
    plt.legend()
    plt.title(f'Sample {sample_idx}, Channel {channel_idx}')
    plt.show()


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
    # 对每个通道的每个样本进行去噪
    denoised_data = np.zeros_like(all_data)
    for i in range(all_data.shape[0]):  # 遍历所有样本
        for j in range(all_data.shape[1]):  # 遍历所有通道
            denoised_data[i, j] = denoise_signal(all_data[i, j])

    # 使用去噪后的数据替换原始数据
    all_data = denoised_data
    
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
    # torch.save(embeddings, os.path.join(embedding_dir, 'Y.pt'))

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
    # torch.save(Z, os.path.join(embedding_dir, 'Z_lamda{}_beta{}.pt'.format(args.lambda_, args.beta_)))
    # print('Z saved')
    # np.savez(os.path.join(statistics_dir, 'RHT_lamda{}_beta{}.npz'.format(args.lambda_, args.beta_)), R_list=R,
    #          Cl_list=Cl, anomalies=tau)
    # print('RHT saved')


if __name__ == '__main__':
    main(mode='train', model_path='results/save_model/2024-12-12_18-11-39.pth')
    