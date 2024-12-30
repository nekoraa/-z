import math
import copy
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from pyitcast.transformer_utils import Batch


# from pyitcast.transformer_utils import get_std_opt
# from pyitcast.transformer_utils import LabelSmoothing
# from pyitcast.transformer_utils import SimpleLossCompute
# from pyitcast.transformer_utils import run_epoch
# from pyitcast.transformer_utils import greedy_decode


class 词嵌入模块(nn.Module):
    def __init__(self, 嵌入维度, 词表大小):
        super(词嵌入模块, self).__init__()
        self.嵌入层 = nn.Embedding(词表大小, 嵌入维度)
        self.嵌入维度 = 嵌入维度

    def forward(self, 映射输入):
        return self.嵌入层(映射输入) * math.sqrt(self.嵌入维度)

    @property
    def d_model(self):
        return self.嵌入维度


class 卷积嵌入层(nn.Module):
    def __init__(self, 嵌入维度, 占位, 隐藏层维度=129):
        super(卷积嵌入层, self).__init__()

        # 定义全连接层，将 (通道数 * 频域高度) 映射到隐藏层维度
        self.全连接层1 = nn.Linear(4 * 129, 隐藏层维度)
        self.relu1 = nn.ReLU()  # 激活函数

        # 第二个全连接层，将隐藏层维度映射到嵌入维度
        self.全连接层2 = nn.Linear(隐藏层维度, 嵌入维度)
        self.relu2 = nn.ReLU()  # 激活函数

        # 第三个全连接层 (可选), 将嵌入维度进一步处理
        self.全连接层3 = nn.Linear(嵌入维度, 嵌入维度)
        self.relu3 = nn.ReLU()  # 激活函数

    def forward(self, 输入):
        # print(输入.shape)
        批次大小, 通道数, 频域高度, 时间长 = 输入.shape

        # 保持4D张量形状 (批次大小, 1, 通道数 * 频域高度, 时间长)
        输入 = 输入.view(批次大小, 1, 通道数 * 频域高度, 时间长)

        # 重排维度，将其从 (批次大小, 1, 通道数 * 频域高度, 时间长)
        # 转换为 (批次大小, 时间长, 通道数 * 频域高度)
        输出张量 = 输入.permute(0, 3, 1, 2).reshape(批次大小, 时间长, 通道数 * 频域高度).squeeze(1)

        # 第一层全连接 + 激活函数
        输出张量 = self.relu1(self.全连接层1(输出张量))

        # 第二层全连接 + 激活函数
        输出张量 = self.relu2(self.全连接层2(输出张量))

        # 第三层全连接 + 激活函数 (如果需要进一步处理)
        输出张量 = self.relu3(self.全连接层3(输出张量))

        # print(输出张量.shape)
        return 输出张量


class 位置编码器(nn.Module):
    def __init__(self, 嵌入维度, 置零比率, 句子最大长度=5000):
        super(位置编码器, self).__init__()
        self.置零层 = nn.Dropout(p=置零比率)
        位置编码矩阵 = torch.zeros(句子最大长度, 嵌入维度)
        绝对位置矩阵 = torch.arange(0, 句子最大长度).unsqueeze(1)
        变换矩阵 = torch.exp(torch.arange(0, 嵌入维度, 2) * -(math.log(10000.0) / 嵌入维度))

        位置编码矩阵[:, 0::2] = torch.sin(绝对位置矩阵 * 变换矩阵)
        位置编码矩阵[:, 1::2] = torch.cos(绝对位置矩阵 * 变换矩阵)

        位置编码矩阵 = 位置编码矩阵.unsqueeze(0)
        self.register_buffer("位置编码矩阵", 位置编码矩阵)

    def forward(self, 输入):
        输入 = 输入 + Variable(self.位置编码矩阵[:, :输入.size(1)], requires_grad=False)
        return self.置零层(输入)


def 掩码张量矩阵生成(大小):
    张量 = (1, 大小, 大小)
    掩码矩阵 = np.triu(np.ones(张量), k=1).astype("uint8")
    return torch.from_numpy(1 - 掩码矩阵)


class QKV生成层(nn.Module):
    def __init__(self, 嵌入维度, 置零比率=0.1):
        super(QKV生成层, self).__init__()

        self.Q线性层 = nn.Linear(嵌入维度, 嵌入维度)
        self.K线性层 = nn.Linear(嵌入维度, 嵌入维度)
        self.V线性层 = nn.Linear(嵌入维度, 嵌入维度)
        self.置零层 = nn.Dropout(p=置零比率)

    def forward(self, 输入):
        Q = self.Q线性层(输入)  # [batch_size, sequence_length, 嵌入维度]
        K = self.K线性层(输入)  # [batch_size, sequence_length, 嵌入维度]
        V = self.V线性层(输入)  # [batch_size, sequence_length, 嵌入维度]

        return Q, K, V


def 自注意力(Q, K, V, 掩码=None, 置零层=None):
    词嵌入维度 = Q.size(-1)
    中间值 = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(词嵌入维度)

    if 掩码 is not None:
        中间值 = 中间值.masked_fill(掩码 == 0, -1e9)

    注意力矩阵 = F.softmax(中间值, dim=-1)

    if 置零层 is not None:
        注意力矩阵 = 置零层(注意力矩阵)

    return torch.matmul(注意力矩阵, V), 注意力矩阵


def 克隆(模型, 克隆次数):
    return nn.ModuleList([copy.deepcopy(模型) for _ in range(克隆次数)])


class 多头注意力模块(nn.Module):
    def __init__(self, 头数, 嵌入维度, 置零比率=0.1):
        super(多头注意力模块, self).__init__()

        assert 嵌入维度 % 头数 == 0

        self.拆分后嵌入维度 = 嵌入维度 // 头数
        self.头数 = 头数
        self.嵌入维度 = 嵌入维度

        self.线性层列表 = 克隆(nn.Linear(嵌入维度, 嵌入维度), 4)
        self.注意力张量 = None
        self.置零层 = nn.Dropout(p=置零比率)

    def forward(self, Q, K, V, 掩码=None):
        if 掩码 is not None:
            掩码 = 掩码.unsqueeze(1)

        批次数量 = Q.size(0)

        Q, K, V = \
            [模型(x).view(批次数量, -1, self.头数, self.拆分后嵌入维度).transpose(1, 2)
             for 模型, x in zip(self.线性层列表, (Q, K, V))]

        x, self.注意力张量 = 自注意力(Q, K, V, 掩码=掩码, 置零层=self.置零层)

        x = x.transpose(1, 2).contiguous().view(批次数量, -1, self.头数 * self.拆分后嵌入维度)

        return self.线性层列表[-1](x)


class 前馈神经网络(nn.Module):
    def __init__(self, 嵌入维度, 中间维度, 置零比率=0.1):
        super(前馈神经网络, self).__init__()
        self.w1 = nn.Linear(嵌入维度, 中间维度)
        self.w2 = nn.Linear(中间维度, 嵌入维度)
        self.置零层 = nn.Dropout(置零比率)

    def forward(self, 输入):
        输入1 = self.w1(输入)
        输出 = self.w2(self.置零层(F.relu(输入1)))

        return 输出


class 规范化层(nn.Module):
    def __init__(self, 词嵌入维度, eps=1e-6):
        super(规范化层, self).__init__()

        self.张量1 = nn.Parameter(torch.ones(词嵌入维度))
        self.张量2 = nn.Parameter(torch.zeros(词嵌入维度))

        self.eps = eps

    def forward(self, 输入):
        均值 = 输入.mean(-1, keepdim=True)
        标准差 = 输入.std(-1, keepdim=True)

        return self.张量1 * (输入 - 均值) / (标准差 + self.eps) + self.张量2


class 子层连接(nn.Module):
    def __init__(self, 大小, 置零比率=0.1):
        super(子层连接, self).__init__()

        self.规范化层 = 规范化层(大小)
        self.置零层 = nn.Dropout(p=置零比率)
        self.大小 = 大小

    def forward(self, 输入, 子层函数):
        return 输入 + self.置零层(子层函数(self.规范化层(输入)))


class 编码器层(nn.Module):
    def __init__(self, 嵌入维度, 多头注意力层, 前馈神经网络, 置零比率):
        super(编码器层, self).__init__()
        self.多头注意力层 = 多头注意力层
        self.前馈神经网络 = 前馈神经网络
        self.嵌入维度 = 嵌入维度

        self.子层连接 = 克隆(子层连接(嵌入维度, 置零比率), 2)
        self.QKV生成层 = QKV生成层(嵌入维度, 置零比率)

    def forward(self, 输入, 掩码):
        输入 = self.子层连接[0](输入, lambda 输入:
        self.多头注意力层(self.QKV生成层(输入)[0], self.QKV生成层(输入)[1], self.QKV生成层(输入)[2], 掩码))

        return self.子层连接[1](输入, self.前馈神经网络)


class 解码器层(nn.Module):
    def __init__(self, 嵌入维度, 多头注意力层, 编码解码注意力层, 前馈神经网络, 置零比率):
        super(解码器层, self).__init__()

        self.多头注意力层 = 多头注意力层
        self.前馈神经网络 = 前馈神经网络
        self.嵌入维度 = 嵌入维度
        self.编码解码注意力层 = 编码解码注意力层
        self.子层连接 = 克隆(子层连接(嵌入维度, 置零比率), 3)
        self.QKV生成层 = QKV生成层(嵌入维度, 置零比率)
        self.Q生成层 = QKV生成层(嵌入维度, 置零比率)
        self.KV生成层 = QKV生成层(嵌入维度, 置零比率)

    def forward(self, 输入, 编码器输出, 掩码, 屏蔽词掩码):
        输入 = self.子层连接[0](输入, lambda 输入: self.多头注意力层(self.QKV生成层(输入)[0], self.QKV生成层(输入)[1],
                                                                     self.QKV生成层(输入)[2], 屏蔽词掩码))

        输入 = self.子层连接[1](输入,
                                lambda 输入: self.编码解码注意力层(self.Q生成层(输入)[0], self.KV生成层(编码器输出)[1],
                                                                   self.KV生成层(编码器输出)[2], 掩码))

        return self.子层连接[2](输入, self.前馈神经网络)


class 总解码器(nn.Module):
    def __init__(self, 解码器层, 解码器层层数):
        super(总解码器, self).__init__()
        self.解码器列表 = 克隆(解码器层, 解码器层层数)
        self.规范化层 = 规范化层(解码器层.嵌入维度)

    def forward(self, 输入, 编码器输出, 掩码, 屏蔽词掩码):
        for 解码器 in self.解码器列表:
            输入 = 解码器(输入, 编码器输出, 掩码, 屏蔽词掩码)
        return self.规范化层(输入)


class 总编码器(nn.Module):
    def __init__(self, 编码器层, 编码器层层数):
        super(总编码器, self).__init__()
        self.编码器列表 = 克隆(编码器层, 编码器层层数)
        self.规范化层 = 规范化层(编码器层.嵌入维度)

    def forward(self, 输入, 掩码):
        for 编码器 in self.编码器列表:
            输入 = 编码器(输入, 掩码)
        return self.规范化层(输入)


class 尾层(nn.Module):
    def __init__(self, 嵌入维度, 词表大小):
        super(尾层, self).__init__()
        self.线性层 = nn.Linear(嵌入维度, 词表大小)

    def forward(self, 输入):
        return F.log_softmax(self.线性层(输入), dim=-1)

    # 修改过请注意


class 编码解码器(nn.Module):
    def __init__(self, 总编码器, 总解码器, 卷积嵌入层, 目标数据嵌入函数, 尾层):
        super(编码解码器, self).__init__()

        self.总编码器 = 总编码器
        self.总解码器 = 总解码器
        self.原数据嵌入函数 = 卷积嵌入层
        self.目标数据嵌入函数 = 目标数据嵌入函数
        self.尾层 = 尾层

    def forward(self, 源数据, 目标数据, 掩码, 屏蔽词掩码):
        return self.解码(self.编码(源数据, 掩码), 掩码, 目标数据, 屏蔽词掩码)

        # 修改过请注意

    def 编码(self, 源数据, 掩码):
        return self.总编码器(self.原数据嵌入函数(源数据), 掩码)

    def 解码(self, 编码器输出, 掩码, 目标数据输入, 屏蔽词掩码):
        return self.总解码器(self.目标数据嵌入函数(目标数据输入), 编码器输出, 掩码, 屏蔽词掩码)

    @property
    def src_embed(self):
        return self.原数据嵌入函数

    @property
    def tgt_embed(self):
        return self.目标数据嵌入函数

    @property
    def encode(self):
        return self.编码

    @property
    def decode(self):
        return self.解码

    @property
    def generator(self):
        return self.尾层


def 制作模型(源数据词汇数, 目标数据词汇数, N=6, 词嵌入维度=512, 中间层维度=2048, 头数=8, 置零比率=0.1):
    c = copy.deepcopy
    注意力层 = 多头注意力模块(头数, 词嵌入维度, 置零比率)
    前馈网络 = 前馈神经网络(词嵌入维度, 中间层维度, 置零比率)
    位置编码 = 位置编码器(词嵌入维度, 置零比率)

    模型 = 编码解码器(
        总编码器(编码器层(词嵌入维度, c(注意力层), c(前馈网络), 置零比率), N),
        总解码器(解码器层(词嵌入维度, c(注意力层), c(注意力层), c(前馈网络), 置零比率), N),
        nn.Sequential(卷积嵌入层(词嵌入维度, 源数据词汇数), c(位置编码)),
        nn.Sequential(词嵌入模块(词嵌入维度, 目标数据词汇数), c(位置编码)),
        尾层(词嵌入维度, 目标数据词汇数)
    )

    for p in 模型.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return 模型


# 模型 = 制作模型(1000, 1000, N=6, 词嵌入维度=2048, 中间层维度=4056, 头数=8, 置零比率=0.1)


def 数据生成器(V, 批次大小, 输送轮数):
    for i in range(输送轮数):
        随机整数 = np.random.randint(0, 11)
        随机数 = 0.1 * 随机整数

        # 使用扩展方法生成一个形状为 (批次大小, 10, 3, 496, 369) 的张量
        生成的张量 = torch.full((批次大小, 10, 3, 496, 369), 随机数)

        数据 = torch.from_numpy(np.random.randint(1, V, size=(批次大小, 10))).long()
        数据[:, 0] = 1

        源数据 = Variable(数据, requires_grad=False)
        源数据 = 源数据.to("cuda")
        目标数据 = Variable(数据, requires_grad=False)
        目标数据 = 目标数据.to("cuda")

        yield Batch(源数据, 目标数据)


