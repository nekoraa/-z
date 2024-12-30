# import random
#
# import torch
#
# # 创建一个示例多维张量
# tensor = torch.randn(4, 10, 3, 496, 368)  # 示例张量，形状为 (2, 3, 4)
#
# # 取出最后一维的最后一个元素
# last_element = float(tensor[..., -1][-1][-1][-1][-1])
#
# print("原始张量:")
# print(tensor)
# print("\n最后一维的最后一个元素:")
# print(last_element)
#
# 随机数1 = random.randint(0, 25)
# 随机数2 = random.randint(0, 25)
# 随机数3 = random.randint(0, 25)
# 随机数4 = random.randint(0, 25)
# 随机数5 = random.randint(0, 25)
# 源数据1 = torch.full([3, 496, 369], 随机数1 / 10)
# 源数据2 = torch.full([3, 496, 369], 随机数2 / 10)
# 源数据3 = torch.full([3, 496, 369], 随机数3 / 10)
# 源数据4 = torch.full([3, 496, 369], 随机数4 / 10)
# 源数据5 = torch.full([3, 496, 369], 随机数5 / 10)
#
# 源数据 = torch.stack([源数据1, 源数据2, 源数据3, 源数据4, 源数据5])
#
# 目标数据 = torch.tensor([0, 随机数1, 随机数2, 随机数3, 随机数4, 随机数5, 27])
#
# print(源数据)
# print(目标数据)
import os

import torch
from torch import nn


def convert_to_log_probs(tensor, vocab_size, padding_idx=0):
    """
    将形状为 (批次大小, 序列长度) 的张量转换为形状为 (批次大小, 序列长度, 目标词汇数) 的对数概率分布张量

    参数：
    tensor (torch.Tensor): 形状为 (批次大小, 序列长度) 的张量，包含类别索引
    vocab_size (int): 目标词汇表大小
    padding_idx (int): 用于填充的索引（默认为0）

    返回：
    torch.Tensor: 形状为 (批次大小, 序列长度, 目标词汇数) 的对数概率分布张量
    """
    batch_size, seq_len = tensor.size()

    # 创建形状为 (批次大小, 序列长度, 目标词汇数) 的张量，初始值为负无穷
    log_probs = torch.full((batch_size, seq_len, vocab_size), -float('inf'))

    # 将目标类别的对数概率设置为0.0
    for i in range(batch_size):
        for j in range(seq_len):
            if tensor[i, j] != padding_idx:
                log_probs[i, j, tensor[i, j]] = 0.0

    return log_probs


def label_smoothing_log_probs(log_probs, alpha=0.1):
    """
    对形状为 (批次大小, 序列长度, 目标词汇数) 的对数概率分布进行标签平滑。

    参数:
        log_probs (torch.Tensor): 对数概率分布张量，形状为 (批次大小, 序列长度, 目标词汇数)
        alpha (float): 平滑参数，默认值为 0.1

    返回:
        torch.Tensor: 平滑后的对数概率分布，形状为 (批次大小, 序列长度, 目标词汇数)
    """
    vocab_size = log_probs.size(-1)

    # 将对数概率分布转换为概率分布
    probs = log_probs.exp()

    # 进行标签平滑
    smooth_probs = (1 - alpha) * probs + alpha / vocab_size

    # 再次转换为对数概率分布
    smooth_log_probs = smooth_probs.log()

    return smooth_log_probs


def label_smoothing(probs, alpha=0.1):
    """
    对形状为 (批次大小, 序列长度, 目标词汇数) 的概率分布进行标签平滑。

    参数:
        probs (torch.Tensor): 概率分布张量，形状为 (批次大小, 序列长度, 目标词汇数)
        alpha (float): 平滑参数，默认值为 0.1

    返回:
        torch.Tensor: 平滑后的概率分布，形状为 (批次大小, 序列长度, 目标词汇数)
    """
    K = probs.size(-1)
    smooth_probs = (1 - alpha) * probs + alpha / K
    return smooth_probs


def get_nth_file_in_folder(folder_path, n):
    # 获取文件夹下的所有文件
    files = os.listdir(folder_path)

    # 按照文件名排序（如果需要）
    files.sort()  # 可以根据需要排序，例如按照文件名字典顺序

    # 选择第n个文件
    if n < len(files):
        file_to_read = os.path.join(folder_path, files[n])
        return file_to_read
    else:
        return None


def get_nth_folder_path(main_folder_path, n):
    # 获取主文件夹下的所有文件夹
    folders = [folder for folder in os.listdir(main_folder_path) if
               os.path.isdir(os.path.join(main_folder_path, folder))]

    # 按照文件夹名称排序（如果需要）
    folders.sort()  # 可以根据需要排序，例如按照文件夹名称的字典顺序

    # 返回第n个文件夹的路径
    if n < len(folders):
        return os.path.join(main_folder_path, folders[n])
    else:
        return None  # 如果n超出了文件夹列表的范围，返回None或者适当的错误处理


def get_nth_folder_name(main_folder_path, n):
    # 获取主文件夹下的所有文件夹
    folders = [folder for folder in os.listdir(main_folder_path) if
               os.path.isdir(os.path.join(main_folder_path, folder))]

    # 按照文件夹名称排序
    folders.sort()  # 可以根据需要排序，例如按照文件夹名称的字典顺序

    # 返回第n个文件夹的名称
    if n < len(folders):
        return folders[n]
    else:
        return None  # 如果n超出了文件夹列表的范围，返回None或者适当的错误处理


def get_subfolder_count(folder_path):
    # 获取文件夹下所有的子文件夹
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    # 返回子文件夹的数量
    return len(subfolders)
