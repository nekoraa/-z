import socket
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, iirnotch, butter
from scipy.fft import fft, fftfreq
import numpy as np
from scipy.signal import welch
import numpy as np
from sklearn.decomposition import FastICA
import random
import os
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
import random
import string
from scipy.signal import spectrogram

def 低通滤波器(信号):
    采样频率 = 500  # 设置采样频率 (Hz)
    截止频率 = 100  # 低通滤波器的截止频率 (Hz)
    阶数 = 4  # 滤波器的阶数
    b, a = butter(阶数, 截止频率 / (0.5 * 采样频率), btype='low')
    # 数据偏移，将 32768 作为“0”点

    滤波后信号 = filtfilt(b, a, 信号)

    # 恢复偏移

    return 滤波后信号
    return 滤波后信号


# 高通滤波器
def 高通滤波器(信号):
    采样频率 = 500  # 设置采样频率 (Hz)
    截止频率 = 10  # 高通滤波器的截止频率 (Hz)
    阶数 = 4  # 滤波器的阶数
    b, a = butter(阶数, 截止频率 / (0.5 * 采样频率), btype='high')
    # 数据偏移，将 32768 作为“0”点

    滤波后信号 = filtfilt(b, a, 信号)

    # 恢复偏移

    return 滤波后信号
    return 滤波后信号


# 带通滤波器
def 带通滤波器(信号):
    采样频率 = 500  # 设置采样频率 (Hz)
    低截止频率 = 1  # 带通滤波器的低截止频率 (Hz)
    高截止频率 = 200 # 带通滤波器的高截止频率 (Hz)
    阶数 = 4  # 滤波器的阶数
    b, a = butter(阶数, [低截止频率 / (0.5 * 采样频率), 高截止频率 / (0.5 * 采样频率)], btype='band')
    # 数据偏移，将 32768 作为“0”点

    滤波后信号 = filtfilt(b, a, 信号)

    return 滤波后信号
    return 滤波后信号


# 带阻滤波器
def 带阻滤波器(信号):
    采样频率 = 500  # 设置采样频率 (Hz)
    低截止频率 = 49  # 带阻滤波器的低截止频率 (Hz)
    高截止频率 = 51  # 带阻滤波器的高截止频率 (Hz)
    阶数 = 4  # 滤波器的阶数
    b, a = butter(阶数, [低截止频率 / (0.5 * 采样频率), 高截止频率 / (0.5 * 采样频率)], btype='bandstop')
    # 数据偏移，将 32768 作为“0”点

    滤波后信号 = filtfilt(b, a, 信号)

    # 恢复偏移

    return 滤波后信号


def 带阻滤波器2(信号):
    采样频率 = 500  # 设置采样频率 (Hz)
    低截止频率 = 99  # 带阻滤波器的低截止频率 (Hz)
    高截止频率 = 101  # 带阻滤波器的高截止频率 (Hz)
    阶数 = 4  # 滤波器的阶数
    b, a = butter(阶数, [低截止频率 / (0.5 * 采样频率), 高截止频率 / (0.5 * 采样频率)], btype='bandstop')
    # 数据偏移，将 32768 作为“0”点
    滤波后信号 = filtfilt(b, a, 信号)
    # 恢复偏移
    return 滤波后信号
    return 滤波后信号


def 独立成分分析(eeg_data, n_components=4, random_state=0):
    """
    使用独立成分分析(ICA)去除脑电信号中的伪迹
    :param eeg_data: 输入的二维列表，形状为 (4, 500)
    :param n_components: ICA 分解后的独立成分数目
    :param random_state: 随机种子，确保每次结果一致
    :return: 去除伪迹后的二维列表，形状与输入相同
    """

    # 将输入的二维列表转换为 numpy 数组
    eeg_data = np.array(eeg_data)

    # 转置数据，使其成为 (500, 4) 的形状，便于 ICA 处理
    eeg_data_transposed = eeg_data.T

    # 创建 FastICA 模型
    ica = FastICA(n_components=n_components, random_state=random_state)

    # 进行 ICA 分解
    sources = ica.fit_transform(eeg_data_transposed)

    # 还原去伪迹后的信号
    eeg_cleaned = ica.inverse_transform(sources)

    # 将结果转回原来的形状 (4, 500)
    eeg_cleaned = eeg_cleaned.T

    # 将输出的 numpy 数组转换回二维列表
    eeg_cleaned_list = eeg_cleaned.tolist()

    return eeg_cleaned_list


def 查看列表形状(eeg_data):
    # 查看列表的总行数（通道数）
    rows = len(eeg_data)

    # 查看每一行的长度
    columns_per_row = [len(row) for row in eeg_data]

    # 打印每个通道的长度
    print(f"总共有 {rows} 个通道，每个通道的数据长度分别为: {columns_per_row}")

    # 输出整体形状信息
    return rows, columns_per_row


def 随机生成数字序列():
    # 序列长度 = random.randint(1, 3)  # 随机选择序列长度为3到5
    序列长度 = 1
    数字序列 = [random.randint(1, 4) for _ in range(序列长度)]  # 随机生成1到9之间的数字序列
    return 数字序列


# 生成随机的10位自然数序列
def 随机生成文件夹名():
    return ''.join([str(random.randint(0, 9)) for _ in range(10)])


# 生成时频图并保存为图片
def 生成时频图(信号, 通道号, 切片号, 文件夹路径, 采样率=500):
    # 计算时频图
    f, t, Sxx = spectrogram(信号, fs=采样率, nperseg=64)

    # 绘制时频图
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')

    # 关闭坐标轴
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    # 保存图片
    图片路径 = os.path.join(文件夹路径, f'{切片号}.jpg')
    plt.savefig(图片路径, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()


# 主函数
def 处理采样信号(采样信号, 数字序列, 采样率=500):
    print(数字序列)
    # 创建"训练数据"文件夹
    根文件夹 = "训练数据"
    if not os.path.exists(根文件夹):
        os.makedirs(根文件夹)

    # 生成随机的10位数字作为文件夹名称
    随机文件夹名 = 随机生成文件夹名()
    文件夹路径 = os.path.join(根文件夹, 随机文件夹名)
    os.makedirs(文件夹路径)

    # 遍历每个通道
    通道数 = len(采样信号)
    for 通道号 in range(通道数):
        通道数据 = 采样信号[通道号]
        子文件夹名 = f'{数字序列}{通道号 + 1}通道'
        子文件夹路径 = os.path.join(文件夹路径, 子文件夹名)
        os.makedirs(子文件夹路径)

        # 对信号进行滤波
        通道数据 = 带通滤波器(通道数据)
        通道数据 = 带阻滤波器(通道数据)
        通道数据 = 带阻滤波器2(通道数据)

        # 生成时频数组
        时频数组 = 生成时频数组(通道数据, 采样率)

        # 保存时频数组为.npy文件
        文件名 = f"时频数组_通道{通道号 + 1}.npy"
        文件路径 = os.path.join(子文件夹路径, 文件名)
        np.save(文件路径, 时频数组)


def 生成时频数组(信号, 采样率):
    """
    根据输入信号生成时频数组（例如短时傅里叶变换）。
    参数：
        信号: 一维时间序列数据
        采样率: 信号采样率
    返回：
        时频数组: 二维时频表示
    """
    from scipy.signal import spectrogram
    频率, 时间, 时频强度 = spectrogram(信号, fs=采样率, nperseg=128, noverlap=64)
    return 时频强度


# # 示例使用
# 采样信号 = [
#     np.random.randn(800),  # 通道1，800个采样点
#     np.random.randn(1000),  # 通道2，1000个采样点
#     np.random.randn(900),  # 通道3，900个采样点
#     np.random.randn(750)  # 通道4，750个采样点
# ]
#
# 数字序列 = "352"
# 处理采样信号(采样信号, 数字序列)

def 检查数组长度一致性(录制数据列表):
    # 获取第一个数组的长度作为参考
    if len(录制数据列表) == 0:
        return True  # 如果列表为空，认为长度一致

    参考长度 = len(录制数据列表[0])

    # 检查所有数组的长度是否与参考长度一致
    for 数组 in 录制数据列表:
        if len(数组) != 参考长度:
            return False

    return True





def create_spectrogram_images(samples, num_sequence):
    def save_spectrogram_image(spectrogram_data, file_path):
        plt.imshow(spectrogram_data, aspect='auto', origin='lower')
        plt.axis('off')
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def create_folder_structure(base_path, num_sequence):
        random_folder = ''.join(random.choices(string.digits, k=10))
        base_folder = os.path.join(base_path, random_folder)
        os.makedirs(base_folder, exist_ok=True)

        for channel in range(4):
            channel_folder = os.path.join(base_folder, f'{num_sequence}第{channel + 1}通道')
            os.makedirs(channel_folder, exist_ok=True)
        return base_folder

    def process_channel_data(channel_data, base_folder, num_sequence):
        num_samples = 250
        step_size = 50
        num_slices = (len(channel_data) - num_samples) // step_size + 1

        slices = []
        for i in range(num_slices):
            start = i * step_size
            end = start + num_samples
            slice_data = channel_data[start:end]
            if len(slice_data) < num_samples:
                slice_data = np.pad(slice_data, (0, num_samples - len(slice_data)), mode='constant')
            slices.append(slice_data)

        # Save individual spectrograms
        for idx, slice_data in enumerate(slices):
            f, t, Sxx = spectrogram(slice_data, fs=500)
            spectrogram_image = np.log(Sxx + 1e-8)  # Avoid log(0)
            file_path = os.path.join(base_folder, f'{num_sequence}第{channel + 1}通道', f'{idx + 1}.jpg')
            save_spectrogram_image(spectrogram_image, file_path)

        # Save total spectrogram
        total_data = np.concatenate(slices)
        f, t, Sxx = spectrogram(total_data, fs=500)
        spectrogram_image = np.log(Sxx + 1e-8)  # Avoid log(0)
        total_file_path = os.path.join(base_folder, f'{num_sequence}第{channel + 1}通道',
                                       f'第{channel + 1}通道时频图.jpg')
        save_spectrogram_image(spectrogram_image, total_file_path)

    base_folder = create_folder_structure('训练数据', num_sequence)

    for channel in range(4):
        process_channel_data(samples[channel], base_folder, num_sequence)


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import random


def 生成文件夹(根目录):
    # 随机生成10位数字序列作为文件夹名
    文件夹名 = ''.join([str(random.randint(0, 9)) for _ in range(10)])
    文件夹路径 = os.path.join(根目录, 文件夹名)
    os.makedirs(文件夹路径)
    return 文件夹路径


def 生成时频图(信号, 采样率=500):
    # 使用STFT生成时频图
    f, t, Zxx = stft(信号, fs=采样率, nperseg=250, noverlap=200)
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.axis('off')  # 移除坐标轴
    plt.tight_layout()


def 处理通道(信号, 数字序列, 通道号, 目录路径):
    通道文件夹 = os.path.join(目录路径, f"{数字序列}第{通道号}通道")
    os.makedirs(通道文件夹)

    # 切片信号，每个250点，步长50，不足补0
    切片列表 = []
    for i in range(0, len(信号), 50):
        切片 = 信号[i:i + 250]
        if len(切片) < 250:
            切片 = np.pad(切片, (0, 250 - len(切片)), 'constant')
        切片列表.append(切片)

    # 生成切片时频图
    for idx, 切片 in enumerate(切片列表):
        生成时频图(切片)
        plt.savefig(os.path.join(通道文件夹, f"{idx + 1}.jpg"), bbox_inches='tight', pad_inches=0)
        plt.close()

    # 生成总时频图
    生成时频图(np.concatenate(切片列表))
    plt.savefig(os.path.join(通道文件夹, f"第{通道号}通道时频图.jpg"), bbox_inches='tight', pad_inches=0)
    plt.close()
