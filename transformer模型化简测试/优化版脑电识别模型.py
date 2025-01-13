import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import 保存训练数据测试
import torch.nn.functional as F

# 模型参数
最大序列长度 = 1000
序列长度 = 3
嵌入维度 = 256
注意头数 = 8
隐藏层大小 = 512
批量大小 = 1
学习率 = 0.0001
训练轮数 = 10
丢弃率 = 0.1

def 归一化(input_tensor, num_classes=28):
    """
    将形状为 (批次大小, 6) 的张量扩展为形状为 (批次大小, 6, 28) 的张量，
    最后一维的元素根据输入张量中的值进行索引设置为1，其余位置为0。

    参数:
    - input_tensor (Tensor): 输入张量，形状为 (批次大小, 6)。
    - num_classes (int): 输出张量的最后一维大小，默认为28。

    返回:
    - output_tensor (Tensor): 扩展后的张量，形状为 (批次大小, 6, num_classes)。
    """
    # 确保输入张量的元素是整数类型
    input_tensor = input_tensor.long()

    # 创建输出张量，形状为 (批次大小, 6, num_classes)，并初始化为0
    output_tensor = torch.zeros(input_tensor.size(0), input_tensor.size(1), num_classes)

    # 使用 long 类型的索引
    output_tensor[torch.arange(input_tensor.size(0)).unsqueeze(1),
                  torch.arange(input_tensor.size(1)).unsqueeze(0),
                  input_tensor] = 1

    return output_tensor

# 数据生成函数
def 生成数据(批量大小, 序列长度):
    输入 = torch.randint(1, 10, (批量大小, 序列长度))  # 避免填充位置为 0
    输出 = 输入.clone()
    return 输入, 输出


# 掩码生成函数
def 生成掩码(序列长度):
    掩码 = torch.triu(torch.ones(序列长度, 序列长度), diagonal=1)  # 上三角矩阵
    掩码 = 掩码.masked_fill(掩码 == 1, float('-inf'))  # 将上三角部分设为 -inf
    return 掩码


# Transformer 模型
class 数字复制模型(nn.Module):
    def __init__(self, 嵌入维度, 注意头数, 隐藏层大小, 最大序列长度, 输入词汇大小, 输出词汇大小, 丢弃率,
                 隐藏层维度=129):
        super(数字复制模型, self).__init__()

        self.最大序列长度 = 最大序列长度  # 最大序列长度参数
        self.输入嵌入层 = nn.Embedding(输入词汇大小, 嵌入维度)
        self.输出嵌入层 = nn.Embedding(输出词汇大小, 嵌入维度)

        # 定义全连接层，将 (通道数 * 频域高度) 映射到隐藏层维度
        self.全连接层1 = nn.Linear(4 * 129, 隐藏层维度)
        self.relu1 = nn.ReLU()  # 激活函数

        # 第二个全连接层，将隐藏层维度映射到嵌入维度
        self.全连接层2 = nn.Linear(隐藏层维度, 嵌入维度)
        self.relu2 = nn.ReLU()  # 激活函数

        # 第三个全连接层 (可选), 将嵌入维度进一步处理
        self.全连接层3 = nn.Linear(嵌入维度, 嵌入维度)
        self.relu3 = nn.ReLU()  # 激活函数

        # 将位置编码的大小设置为最大序列长度
        self.位置编码 = nn.Parameter(torch.randn(最大序列长度, 嵌入维度))
        self.丢弃层 = nn.Dropout(丢弃率)

        self.变换器 = nn.Transformer(
            d_model=嵌入维度,
            nhead=注意头数,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=隐藏层大小,
            dropout=丢弃率,
        )

        self.输出层 = nn.Linear(嵌入维度, 输出词汇大小)

    def forward(self, 输入序列, 输出序列, src_mask=None, tgt_mask=None):
        # 检查输入和目标序列长度
        if 输入序列.size(1) > self.最大序列长度 or 输出序列.size(1) > self.最大序列长度:
            raise ValueError(f"序列长度超过最大限制 {self.最大序列长度}")

        批次大小, 通道数, 频域高度, 时间长 = 输入序列.shape

        # 保持4D张量形状 (批次大小, 1, 通道数 * 频域高度, 时间长)
        输入 = 输入序列.view(批次大小, 1, 通道数 * 频域高度, 时间长)

        # 重排维度，将其从 (批次大小, 1, 通道数 * 频域高度, 时间长)
        # 转换为 (批次大小, 时间长, 通道数 * 频域高度)
        输出张量 = 输入.permute(0, 3, 1, 2).reshape(批次大小, 时间长, 通道数 * 频域高度).squeeze(1)

        # 第一层全连接 + 激活函数
        输出张量 = self.relu1(self.全连接层1(输出张量))

        # 第二层全连接 + 激活函数
        输出张量 = self.relu2(self.全连接层2(输出张量))

        # 第三层全连接 + 激活函数 (如果需要进一步处理)
        输出张量 = self.relu3(self.全连接层3(输出张量))
        # 输入嵌入 + 位置编码
        输入嵌入 = 输出张量
        输出嵌入 = self.丢弃层(self.输出嵌入层(输出序列) + self.位置编码[:输出序列.size(1)])

        # 使用变换器进行编码和解码
        变换器输出 = self.变换器(
            输入嵌入.permute(1, 0, 2),
            输出嵌入.permute(1, 0, 2),
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )

        # 输出预测
        输出预测 = self.输出层(变换器输出.permute(1, 0, 2))
        return 输出预测


class 数据加载器(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        # 获取文件夹下的所有文件和文件夹
        条目 = os.listdir("训练数据")

        # 过滤出子文件夹，使用 os.path.isdir 判断是否为文件夹
        子文件夹 = [项 for 项 in 条目 if os.path.isdir(os.path.join("训练数据", 项))]

        # 返回子文件夹的数量
        return len(子文件夹)

    def __getitem__(self, item):
        源数据, 目标数据 = 保存训练数据测试.验证时频数组(item)

        return 源数据, 目标数据

train_loader = DataLoader(数据加载器(), batch_size=批量大小, shuffle=True, drop_last=True)

# 初始化模型
模型 = 数字复制模型(嵌入维度, 注意头数, 隐藏层大小, 最大序列长度, 10, 10, 丢弃率)
损失函数 = nn.CrossEntropyLoss()
优化器 = optim.Adam(模型.parameters(), lr=学习率)

src_mask = None  # 编码器不需要掩码



# 训练
for epoch in range(训练轮数):
    running_loss = 0.0

    # 使用 tqdm 封装 train_loader，并设置 desc 和 leave 参数以显示进度条和保留进度条
    with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{训练轮数}', leave=True, ncols=100) as pbar:
        for i, data in enumerate(pbar):
            模型.train()
            源数据, 目标数据 = data

            输出占位 = 目标数据[:, :-1]  # 目标序列从第二个时间步开始（错开一位）
            目标下标 = 目标数据[:, 1:]  # 目标序列从第二个时间步开始（不包括第一个时间步）
            tgt_mask = 生成掩码(输出占位.size(-1))
            优化器.zero_grad()

            目标数据输出概率分布 = 归一化(目标下标, 10).cuda()

            # 使用掩码进行预测
            输出预测 = 模型(源数据, 输出占位, src_mask, tgt_mask)

            # 将目标标签转换为one-hot编码
            目标概率分布 = F.one_hot(目标下标, num_classes=10).float()

            优化器.zero_grad()

            # 对模型输出进行softmax，得到预测的概率分布
            输出预测概率分布 = F.softmax(输出预测, dim=-1)

            # 计算损失，使用目标的概率分布与预测的概率分布进行对比
            损失 = -torch.sum(目标概率分布 * torch.log(输出预测概率分布 + 1e-10), dim=-1).mean()

            损失.backward()
            优化器.step()

            pbar.set_postfix({'Loss': f'{损失.item():.4f}'})



def 模型测试函数(输入, 允许生成最大序列长度=10):
    目标输出 = torch.tensor([[0]])

    for i in range(允许生成最大序列长度 - 1):  # 迭代生成序列，直到达到最大序列长度
        # 获取掩码参数和掩码
        掩码参数 = 目标输出.size(-1)
        掩码 = 生成掩码(掩码参数)

        # 使用模型预测下一个时间步
        测试预测 = 模型(输入, 目标输出, None, 掩码).argmax(dim=-1)
        最后一步预测 = 测试预测[:, -1]

        # 更新目标输出
        目标输出 = torch.cat([目标输出, 最后一步预测.unsqueeze(-1)], dim=-1)
        print(目标输出)
    return 目标输出


# 测试
源数据, 目标数据 = 数据加载器()[0]
测试输入 = 源数据.unsqueeze(0)
测试预测 = 模型测试函数(测试输入, 序列长度)
最后一步预测 = 测试预测[:, -1]  # 取最后一个时间步的输出

print("结果:", 目标数据)
print("预测:", 测试预测)
