from datetime import datetime
from thop import profile as thop_profile  # 更改导入方式
import time
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from utils.loss import ContrastiveLoss


class SiameseDataset(Dataset):
    def __init__(self, pairs, targets):
        self.pairs = pairs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img1, img2 = self.pairs[idx]
        target = self.targets[idx]
        return (
            img1, img2, target
        )


def split_dataset(pairs, targets, test_size=0.2, random_state=42):
    train_pairs, test_pairs, train_targets, test_targets = train_test_split(
        pairs, targets, test_size=test_size, random_state=random_state
    )
    return train_pairs, test_pairs, train_targets, test_targets


def train_siamese_network(model, pairs, targets, test_pairs, test_targets, epochs=100, batch_size=32, lr=0.001,
                          device='cuda', num_workers=1):
    # 创建数据集和加载器
    train_dataset = SiameseDataset(pairs, targets)
    test_dataset = SiameseDataset(test_pairs, test_targets)
    # input_sample1 = torch.randn(1, *pairs[0][0].shape).to(device)
    # input_sample2 = torch.randn(1, *pairs[0][1].shape).to(device)
    # flops, params = thop_profile(model, inputs=(input_sample1, input_sample2))
    # print(f'模型FLOPS: {flops/1e9:.2f}G')
    # print(f'模型参数量: {params/1e6:.2f}M')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    try:
        # Training loop
        model.train()
        print('Start training...')
        for epoch in range(epochs):
            start_time = time.time()  # 记录每个epoch的开始
            if epoch + 1 == 10:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
                print(f"Learning rate adjusted to {optimizer.param_groups[0]['lr']:.6f}")
            if epoch + 1 == 20:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
                print(f"Learning rate adjusted to {optimizer.param_groups[0]['lr']:.6f}")

            epoch_loss = 0.0
            for img1, img2, label in train_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)

                # Forward pass
                output1, output2 = model(img1, img2)

                # Compute loss
                loss = criterion(output1, output2, label)
                epoch_loss += loss.item()
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # end_time = time.time()
            # epoch_time = end_time - start_time
            # samples_per_sec = len(train_loader.dataset) / epoch_time
            # actual_flops = flops * samples_per_sec
            
            # print(f"Epoch [{epoch + 1}/{epochs}]")
            # print(f"训练损失: {epoch_loss / len(train_loader):.4f}")
            # print(f"测试损失: {test_loss / len(test_loader):.4f}")
            # print(f"处理速度: {samples_per_sec:.2f} 样本/秒")
            # print(f"实际FLOPS: {actual_flops/1e9:.2f}G FLOPS")
            # 验证阶段
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for img1, img2, label in test_loader:
                    img1, img2, label = img1.to(device), img2.to(device), label.to(device)

                    # Forward pass
                    output1, output2 = model(img1, img2)

                    # Compute loss
                    loss = criterion(output1, output2, label)
                    test_loss += loss.item()

            # 打印每轮的训练和测试损失
            print(
                f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss / len(train_loader):.4f}, Test Loss: {test_loss / len(test_loader):.4f}")
            model.train()  # 回到训练模式
        torch.save(model.state_dict(), 'results/save_model/{}.pth'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        print("Current model saved.")

    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
        torch.save(model.state_dict(), 'results/save_model/{}.pth'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        print("Current model saved.")

    return model
