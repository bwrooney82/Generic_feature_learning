import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange

# 设置文件夹路径
# folder_path = r"F:\data\Machine\Motor\6M" # xx 文件夹路径
# output_file = "Combined_67M.csv"  # 汇总后的文件名
#
# csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
#
# # 存储拼接结果
# merged_data = pd.DataFrame()
#
#
# row_ranges = [range(500, 1401),range(1500,3000)]
# row_indices = sorted([i for r in row_ranges for i in r])  # 合并所有行号
#
# for file in csv_files:
#     file_path = os.path.join(folder_path, file)
#     file_name = os.path.splitext(file)[0]  # 获取文件名（无扩展名）
#
#     # 读取指定范围的行
#     df = pd.read_csv(file_path, header=None, skiprows=lambda x: x not in row_indices)
#     df.reset_index(drop=True, inplace=True)  # 重置索引，方便后续处理
#
#     # 展平为一列数据（按列拼接）
#     df_flattened = df.stack().reset_index(drop=True)  # 堆叠为单列
#
#     # 添加到汇总表中
#     merged_data[file_name] = df_flattened
#
# # 均值-标准差归一化
# normalized_data = (merged_data - merged_data.mean()) / merged_data.std()
#
# # 保存处理后的数据到新的 CSV 文件
# normalized_data.to_csv(output_file, index=False)
#
# print(f"指定行范围读取并归一化完成，结果已保存到 {output_file}")
#
# # 绘图部分
# num_columns = normalized_data.shape[1]
# fig, axes = plt.subplots(num_columns, 1, figsize=(8, 4 * num_columns), constrained_layout=True)
#
# # 遍历每一列数据进行绘图
# for i, column in enumerate(normalized_data.columns):
#     axes[i].plot(normalized_data[column], label=column, color="blue", linewidth=1)
#     axes[i].set_title(f"{column} (Normalized)", fontsize=12)
#     axes[i].legend(loc="upper right")
#     axes[i].grid(True)
#
# # 显示图形
# plt.show()


def generate_windows(df, column, output_dir, window_size=2048):
    os.makedirs(output_dir, exist_ok=True)

    data = []
    for col in column:
        signal = df[col].values
        num_windows = len(signal) // window_size
        segments = []
        for i in trange(num_windows):
            segment = signal[i * window_size:(i + 1) * window_size]
            segments.append(segment)
        segments = np.array(segments)
        data.append(segments)
    data = np.array(data)[:,:10000,:]
    print(data.shape)
    np.save(os.path.join(output_dir, 'data64.npy'), data)


df = pd.read_csv(r"F:\data\Machine\Motor\64M\Combined_64M.csv")
plt.plot(df['64M_02'])
plt.show()
# generate_windows(df, ['64M_01', '64M_02'], '../data')