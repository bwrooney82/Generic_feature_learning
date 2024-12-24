import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.stats import kurtosis, chi2
from scipy.linalg import inv
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def mardia_test(data):
    n, d = data.shape
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    cov_inv = inv(cov)

    # 偏度检验
    skew_stat = 0
    for i in range(n):
        diff = data[i] - mean
        skew_stat += (diff @ cov_inv @ diff.T) ** 3
    skew_stat /= n

    # 偏度统计量及p值
    skew_chi2 = n * skew_stat / 6
    skew_p = 1 - chi2.cdf(skew_chi2, df=d * (d + 1) * (d + 2) // 6)

    # 峰度检验
    kurt_stat = kurtosis(np.dot(data - mean, cov_inv), fisher=False, axis=0)
    kurt_stat = np.sum(kurt_stat) / d

    # 峰度统计量及p值
    kurt_z = (kurt_stat - d * (d + 2)) / np.sqrt(8 * d * (d + 2) / n)
    kurt_p = 2 * (1 - chi2.cdf(kurt_z ** 2, df=1))

    return skew_stat, skew_p, kurt_stat, kurt_p


embeddings = torch.load('embeddings/2024-11-26_20-58-54.pt').cpu().numpy()
data = embeddings[1200:]
skew_stat, skew_p, kurt_stat, kurt_p = mardia_test(data)
print(f"Skewness Test: stat={skew_stat:.3f}, p={skew_p:.3f}")
print(f"Kurtosis Test: stat={kurt_stat:.3f}, p={kurt_p:.3f}")


# stat, p = multivariate_normality(embeddings[:500], alpha=0.05)
# if p > 0.05:
#     print(f"The data follows a multivariate Gaussian distribution (p={p:.3f})")
# else:
#     print(f"The data does not follow a multivariate Gaussian distribution (p={p:.3f})")

mean = np.mean(data, axis=0)
cov = np.cov(data, rowvar=False)
cov_inv = inv(cov)
mah_distances = [mahalanobis(x, mean, cov_inv) for x in data]

# 绘制直方图
plt.hist(mah_distances, bins=30, density=True)
plt.title("Mahalanobis Distance Distribution")
plt.show()