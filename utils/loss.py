import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive


class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        # 归一化特征向量
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # 拼接正负样本
        z = torch.cat([z_i, z_j], dim=0)  # shape: [2 * batch_size, feature_dim]
        batch_size = z_i.size(0)

        # 计算相似度矩阵
        sim_matrix = torch.mm(z, z.T)  # shape: [2 * batch_size, 2 * batch_size]
        sim_matrix = sim_matrix / self.temperature

        # 正样本相似度
        pos_sim = torch.diag(sim_matrix, batch_size)  # 正样本的相似度（view1 和 view2）
        pos_sim = torch.cat([pos_sim, torch.diag(sim_matrix, -batch_size)])

        # 计算对比损失
        labels = torch.arange(2 * batch_size).to(z.device)  # 目标索引
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
