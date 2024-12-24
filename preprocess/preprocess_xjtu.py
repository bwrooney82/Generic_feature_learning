import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def read_data_as_df(base_dir):
    if base_dir[-1] != '/':
        base_dir += '/'

    dfs = []
    files = sorted(os.listdir(base_dir), key=lambda x: int(os.path.splitext(x)[0]))

    for f in files:
        df = pd.read_csv(base_dir + f)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # normalizing values in the column "Horizontal_vibration_signals"
    hori_mean = combined_df['Horizontal_vibration_signals'].mean()
    hori_sd = combined_df['Horizontal_vibration_signals'].std()
    combined_df['normalized_horizontal'] = (combined_df['Horizontal_vibration_signals'] - hori_mean) / hori_sd

    # normalizing values in the column "Vertical_vibration_signals"
    vert_mean = combined_df['Vertical_vibration_signals'].mean()
    vert_sd = combined_df['Vertical_vibration_signals'].std()
    combined_df['normalized_vertical'] = (combined_df['Vertical_vibration_signals'] - vert_mean) / vert_sd

    return combined_df


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
    data = np.array(data)
    print(data.shape)
    np.save(os.path.join(output_dir, 'data33.npy'), data)


if __name__ == '__main__':
    df = read_data_as_df(r"F:\data\Machine\XJTU-SY\XJTU-SY_Bearing_Datasets\37.5Hz11kN\Bearing2_1")

    df.to_csv('bearing2_1.csv', encoding='utf-8', index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(df['normalized_horizontal'], color='blue')
    axes[0].set_title('Normalized Horizontal Vibration Signals')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Normalized Value')

    axes[1].plot(df['normalized_vertical'], color='green')
    axes[1].set_title('Normalized Vertical Vibration Signals')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Normalized Value')

    plt.tight_layout()
    plt.show()

    df = pd.read_csv(r"F:\data\Machine\XJTU-SY\Processed\bearing3_3.csv")

    generate_windows(df, ['normalized_horizontal', 'normalized_vertical'], '../data')
