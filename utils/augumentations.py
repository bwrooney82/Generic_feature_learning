import numpy as np
import torch


def generate_perturbation(data, noise_level=0.05):
    """
    对正常信号生成随机扰动，模拟伪样本
    Args:
        data (np.ndarray): 原始信号，形状为 (channels, seq_length)
        noise_level (float): 噪声强度比例 (0~1)
    Returns:
        perturbed_data (np.ndarray): 加入噪声后的信号
    """
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


def random_crop(data, crop_ratio=0.9):
    """
    对信号进行随机裁剪
    Args:
        data (np.ndarray): 原始信号，形状为 (channels, seq_length)
        crop_ratio (float): 保留信号比例 (0~1)
    Returns:
        cropped_data (np.ndarray): 裁剪并填充后的信号
    """
    channels, seq_length = data.shape
    crop_length = int(seq_length * crop_ratio)
    start_idx = np.random.randint(0, seq_length - crop_length + 1)
    cropped = data[:, start_idx:start_idx + crop_length]
    # 填充到原始长度
    pad_length = seq_length - crop_length
    padded_data = np.pad(cropped, ((0, 0), (0, pad_length)), mode='constant', constant_values=0)
    return padded_data


def time_warp(data, max_warp=0.1):
    """
    对信号进行随机时序扭曲
    Args:
        data (np.ndarray): 原始信号，形状为 (channels, seq_length)
        max_warp (float): 最大扭曲比例
    Returns:
        warped_data (np.ndarray): 扭曲后的信号
    """
    channels, seq_length = data.shape
    warp_scale = 1 + np.random.uniform(-max_warp, max_warp, seq_length)
    indices = np.cumsum(warp_scale) / np.sum(warp_scale) * seq_length
    indices = np.clip(indices.astype(int), 0, seq_length - 1)
    warped_data = data[:, indices]
    return warped_data

def generate_pairs(normal_data, num_pairs, augmentations=None):
    if augmentations is None:
        augmentations = [generate_perturbation, random_crop, time_warp]

    num_samples, channels, seq_length = normal_data.shape
    pairs = []
    labels = []

    for _ in range(num_pairs):
        idx = np.random.randint(0, num_samples)
        anchor = normal_data[idx]

        augmentation_fn = np.random.choice(augmentations)
        positive = augmentation_fn(anchor)
        pairs.append([anchor, positive])
        labels.append(0)

        neg_idx = np.random.randint(0, num_samples)
        while neg_idx == idx:
            neg_idx = np.random.randint(0, num_samples)
        negative = normal_data[neg_idx]
        pairs.append([anchor, negative])
        labels.append(1)

    pairs = torch.tensor(np.array(pairs), dtype=torch.float32)  # Shape: (num_pairs, 2, channels, seq_length)
    labels = torch.tensor(np.array(labels), dtype=torch.float32)  # Shape: (num_pairs,)
    return pairs, labels