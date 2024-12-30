from torch import nn
from torch.nn import init
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import 神经网络模型
import torch.optim as optim
import torch
import time
import dvre

# ---------------模型创建参数---------------#

词嵌入维度 = 64
中间层维度 = 128
头数 = 8
置零比率 = 0.1
词汇数 = 11

# ---------------训练参数---------------#

训练批次大小 = 10
训练轮数 = 30
初始学习率 = 0.0001


# ---------------主训练---------------#

def 贪婪解码(prob_distributions):
    """
    将模型输出的概率分布按照贪婪解码成类别标签

    参数:
    - prob_distributions: 张量，形状为 (batch_size, 6, 28)，表示模型输出的概率分布

    返回:
    - decoded_labels: 张量，形状为 (batch_size, 6)，表示解码后的类别标签
    """
    # 找到概率分布中的最大值索引
    decoded_labels = torch.argmax(prob_distributions, dim=-1)
    return decoded_labels


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
    # 创建输出张量，形状为 (批次大小, 6, num_classes)，并初始化为0
    output_tensor = torch.zeros(input_tensor.size(0), input_tensor.size(1), num_classes)

    # 将输入张量中的元素值作为索引，将输出张量中对应位置设置为1
    output_tensor[torch.arange(input_tensor.size(0)).unsqueeze(1), torch.arange(input_tensor.size(1)), input_tensor] = 1

    return output_tensor


模型 = 神经网络模型.制作模型(词汇数, 词汇数, N=6, 词嵌入维度=词嵌入维度, 中间层维度=中间层维度, 头数=头数,
                             置零比率=置零比率)


def weights_init_uniform(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.uniform_(m.weight, -0.1, 0.1)  # 均匀初始化权重
        if m.bias is not None:
            init.constant_(m.bias, 0)  # 常数初始化偏置为0


模型.apply(weights_init_uniform)
模型 = 模型.to("cuda")


class 数据加载器(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 1000

    def __getitem__(self, item):
        源数据 = torch.randint(0, 10, (9,))

        # 手动将最后一个元素设置为 10，并扩展为最终的张量
        源数据 = torch.cat((源数据, torch.tensor([10])))
        目标数据 = 源数据
        # print(源数据.dtype)
        return 源数据, 目标数据


train_loader = DataLoader(数据加载器(), batch_size=训练批次大小, shuffle=True, drop_last=True)

# train_loader = DataLoader(训练用函数.process_folders("处理后数据"), batch_size=训练批次大小, shuffle=True, drop_last=True)

criterion = nn.NLLLoss()

# 创建优化器
optimizer = optim.Adam(模型.parameters(), lr=初始学习率)

# 创建StepLR调度器
# scheduler = StepLR(optimizer, step_size=1, gamma=0.8)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

总参数量 = sum(p.numel() for p in 模型.parameters())
可训练参数量 = sum(p.numel() for p in 模型.parameters() if p.requires_grad)

print(f"模型总参数量: {总参数量}")
print(f"可训练参数量: {可训练参数量}")
time.sleep(1)

for epoch in range(训练轮数):
    running_loss = 0.0

    # 使用 tqdm 封装 train_loader，并设置 desc 和 leave 参数以显示进度条和保留进度条
    with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{训练轮数}', leave=True) as pbar:
        for i, data in enumerate(pbar):
            模型.train()
            源数据, 目标数据 = data

            源数据 = 源数据.cuda()
            目标数据输入 = 目标数据[:, :-1].cuda()
            目标数据输出 = 目标数据[:, 1:].cuda()
            token个数 = torch.numel(目标数据输出)

            掩码 = 神经网络模型.掩码张量矩阵生成(目标数据输入.size(-1)).cuda()

            目标数据输出概率分布 = 归一化(目标数据输出, 词汇数).cuda()
            目标数据输出概率分布 = dvre.label_smoothing(目标数据输出概率分布)

            模型输出 = 模型.尾层(模型(源数据, 目标数据输入, None, 掩码))
            损失计算 = nn.KLDivLoss(reduction='sum')
            损失 = 损失计算(模型输出.contiguous().view(-1, 模型输出.size(-1)),
                            目标数据输出概率分布.contiguous().view(-1, 模型输出.size(-1))) / token个数

            损失.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += 损失.item()

            # 更新 tqdm 进度条的显示
            pbar.set_postfix({'Loss': f'{损失.item():.4f}'})

    # 每个 epoch 结束后更新学习率
    print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

    # 保存模型（调整条件避免过早保存）
    if epoch % 10 == 0 or epoch == 训练轮数 - 1:
        torch.save(模型.state_dict(), f'模型_epoch_{epoch + 1}.pth')

    # 打印每个 epoch 的平均损失
    average_loss = running_loss / len(train_loader)
    scheduler.step(average_loss)
    print(f'Epoch [{epoch + 1}/{训练轮数}], Average Loss: {average_loss:.4f}')

torch.save(模型.state_dict(), f'最终模型.pth')
模型.eval()

源数据, 目标数据 = 数据加载器()[0]
源数据 = 源数据.unsqueeze(0)
目标数据输入 = 目标数据


def 模型测试函数(输入):
    # 初始化输入和目标输出
    输入XX = 输入.cuda()
    目标输出 = torch.zeros((1, 1)).cuda().long().cuda()

    for i in range(100):
        # 获取掩码参数和掩码
        掩码参数 = 目标输出.size(-1)
        掩码XX = 神经网络模型.掩码张量矩阵生成(掩码参数).cuda()

        # 通过模型进行前向传播并使用贪婪解码
        输出 = 贪婪解码(模型.尾层(模型(输入XX, 目标输出, None, 掩码XX)))[..., -1]

        # 更新目标输出
        目标输出 = torch.cat((目标输出, 输出.unsqueeze(0)), dim=1).long()

        # 如果输出为27，则退出循环
        if 输出.item() == 10:
            return 目标输出

    return 目标输出


print(模型测试函数(源数据))
print(目标数据输入)
print('训练完成!')


