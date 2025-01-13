import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 模型参数
序列长度 = 10
嵌入维度 = 256
注意头数 = 8
隐藏层大小 = 512
批量大小 = 1
学习率 = 0.0001
训练轮数 = 20
丢弃率 = 0.1


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
    def __init__(self, 嵌入维度, 注意头数, 隐藏层大小, 最大序列长度, 输入词汇大小, 输出词汇大小, 丢弃率):
        super(数字复制模型, self).__init__()

        self.最大序列长度 = 最大序列长度  # 最大序列长度参数
        self.输入嵌入层 = nn.Embedding(输入词汇大小, 嵌入维度)
        self.输出嵌入层 = nn.Embedding(输出词汇大小, 嵌入维度)

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

        # 输入嵌入 + 位置编码
        输入嵌入 = self.丢弃层(self.输入嵌入层(输入序列) + self.位置编码[:输入序列.size(1)])
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


# 初始化模型
模型 = 数字复制模型(嵌入维度, 注意头数, 隐藏层大小, 序列长度, 10, 10, 丢弃率)
损失函数 = nn.CrossEntropyLoss()
优化器 = optim.Adam(模型.parameters(), lr=学习率)

# 掩码
src_mask = None  # 编码器不需要掩码
tgt_mask = 生成掩码(序列长度-1)

class 数据加载器(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 100

    def __getitem__(self, item):
        源数据 = torch.randint(1, 10, (批量大小, 序列长度))  # 避免填充位置为 0
        目标数据 = 源数据.clone()

        return 源数据, 目标数据

train_loader = DataLoader(数据加载器(), batch_size=批量大小, shuffle=True, drop_last=True)

# 训练
for epoch in range(训练轮数):
    running_loss = 0.0

    # 使用 tqdm 封装 train_loader，并设置 desc 和 leave 参数以显示进度条和保留进度条
    with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{训练轮数}', leave=True) as pbar:
        for i, data in enumerate(pbar):
            输入, 目标 = 生成数据(批量大小, 序列长度)

            # 目标序列错开一位
            输出占位 = 目标[:, :-1]  # 目标序列从第二个时间步开始（错开一位）
            目标下标 = 目标[:, 1:]  # 目标序列从第二个时间步开始（不包括第一个时间步）

            # 将目标标签转换为one-hot编码
            目标概率分布 = F.one_hot(目标下标, num_classes=10).float()

            优化器.zero_grad()

            # 获取模型输出，未经过softmax处理
            输出预测 = 模型(输入, 输出占位, src_mask, tgt_mask)

            # 对模型输出进行softmax，得到预测的概率分布
            输出预测概率分布 = F.softmax(输出预测, dim=-1)

            # 计算损失，使用目标的概率分布与预测的概率分布进行对比
            损失 = -torch.sum(目标概率分布 * torch.log(输出预测概率分布 + 1e-10), dim=-1).mean()

            损失.backward()
            优化器.step()
            pbar.set_postfix({'Loss': f'{损失.item():.4f}'})


def 模型测试函数(输入, 最大序列长度=10):
    目标输出 = 输入[:, :1]

    for i in range(最大序列长度 - 1):  # 迭代生成序列，直到达到最大序列长度
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
测试输入, _ = 生成数据(1, 序列长度)
测试预测 = 模型测试函数(测试输入, 序列长度)
最后一步预测 = 测试预测[:, -1]  # 取最后一个时间步的输出
print("输入:", 测试输入)
print("预测:", 测试预测)

