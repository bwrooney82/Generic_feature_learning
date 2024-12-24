import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import trange
from model.cccp import optimize_transformation_matrix


def get_embeddings(model, data, device='cuda', batch_size=100):
    model.eval()
    data = torch.tensor(data, dtype=torch.float32).to(device) if not isinstance(data, torch.Tensor) else data

    # Preallocate embeddings tensor
    num_samples = data.shape[0]
    embedding_dim = model.get_embedding(data[:1]).shape[1]  # Get embedding size
    embeddings = torch.empty((num_samples, embedding_dim), dtype=torch.float32, device=device)

    # Extract embeddings in batches
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_data = data[i:i + batch_size]
            embeddings[i:i + batch_size] = model.get_embedding(batch_data)

    return embeddings


def online_monitoring_statistic(embeddings, start=0, history=10, current=1, lambda_=1.0, beta=0.01,gamma=1e-2, alpha=0.10):
    R_list = []
    Cl_list = []
    anomalies = []

    Y_nor = embeddings[start:start + history]

    print('start monitoring...')
    Z_embeddings = []
    k_history = []
    for i in trange(start, len(embeddings) - history, current):
        Y_cur = embeddings[i + history:i + history + current]
        
        # 优化 A 矩阵
        A,Z_cur, R, cl,k = optimize_transformation_matrix(Y_nor, Y_cur, lambda_=lambda_, beta=beta, max_iter=100,tol=1e-3,gamma=gamma,alpha=alpha)
        # R, cl = RHT_statistic(Z_cur, Z_nor, gamma=gamma, alpha=alpha)
        R_list.append(R)
        Cl_list.append(cl)
        k_history.append(k)
        
        if R > cl:
            anomalies.append((i, R, cl))
        Z_embeddings.append(Z_cur)
    Z_embeddings = torch.cat(Z_embeddings, dim=1).permute(1, 0)
    R_list = np.array(R_list)
    Cl_list = np.array(Cl_list)
    k_history = np.array(k_history)
    np.save('k_history_beta{}_lambda{}.npy'.format(beta,lambda_), k_history)
    print('k_history saved')
    fault_point = None

    for i in range(len(anomalies) - 4):
        # 提取连续个点
        window = anomalies[i:i + 5]

        # 检查窗口中是否所有点的 RHT 超过对应的控制线
        if all(r > cl for _, r, cl in window):
            # 记录第一个早期故障点并退出循环
            fault_point = window[0]  # 记录窗口中第一个点
            break

    if fault_point:
        idx, r, cl = fault_point
        print("First Early Fault Anomaly:")
        print(f"Index: {idx}, RHT: {r:.4f}, CL: {cl:.4f}")
    else:
        print("No early fault anomaly detected.")

    plt.figure(figsize=(10, 6))
    plt.plot(Cl_list, label="Control Limit", color='red', linestyle='--')
    plt.plot(R_list, label="RHT Statistic", color='blue')
    plt.xlabel("Time Steps")
    plt.ylabel("RHT Statistic")
    plt.title("Online Monitoring: RHT Statistic vs Control Limit")
    plt.legend()
    plt.grid(True)
    plt.show()

    return R_list, Cl_list, fault_point[0], Z_embeddings
