import os
import numpy as np
from PyQt6.QtCore import Qt
from scipy.signal import welch
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import socket
import time
import 滤波器
import sys
from PyQt6.QtCore import QCoreApplication

# 绑定的 IP 地址和端口
UDP_IP = "0.0.0.0"  # 监听所有 IP 地址
UDP_PORT = 5005  # 与 ESP32 发送时使用的端口相同

# 创建 UDP 套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print("等待接收数据...")

packet_count = [0] * 4
start_time = [time.time()] * 4
values = [[] for _ in range(4)]
每次处理大小 = 50
采样率 = 500
通道数 = 4
窗口大小 = 2500
频谱图显示大小 = 50

app = QtWidgets.QApplication([])

# 创建主窗口
main_window = QtWidgets.QMainWindow()
main_window.setWindowTitle("脑电处理")
main_window.resize(1200, 800)

# 创建 QSplitter 分割主窗口
splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
main_window.setCentralWidget(splitter)

# 左侧按钮区域
left_widget = QtWidgets.QWidget()
left_layout = QtWidgets.QVBoxLayout()
left_widget.setLayout(left_layout)

# 添加按钮
btn1 = QtWidgets.QPushButton("启动")
btn2 = QtWidgets.QPushButton("暂停")
btn3 = QtWidgets.QPushButton("录制训练数据")
btn4 = QtWidgets.QPushButton("保存训练数据")
btn6 = QtWidgets.QPushButton("取消保存")
btn5 = QtWidgets.QPushButton("清空日志")

left_layout.addWidget(btn1)
left_layout.addWidget(btn2)
left_layout.addWidget(btn3)
left_layout.addWidget(btn4)
left_layout.addWidget(btn5)
left_layout.addWidget(btn6)

btn4.setEnabled(False)
btn4.setStyleSheet("color: gray; background-color: lightgray;")

# 添加日志文本框
log_text_edit = QtWidgets.QTextEdit()
log_text_edit.setReadOnly(True)
log_text_edit.setStyleSheet("background-color: black; color: white;")
left_layout.addWidget(log_text_edit)

# 创建GraphicsLayoutWidget窗口用于波形图
waveform_widget = pg.GraphicsLayoutWidget(show=True, title="8 Waveforms in 2x4 Layout")
waveform_widget.resize(1000, 800)

# 添加左侧和右侧窗口到 QSplitter
splitter.addWidget(left_widget)
splitter.addWidget(waveform_widget)

# 设置右边波形图占比为70%，左边按钮区域占比为30%
splitter.setStretchFactor(0, 30)
splitter.setStretchFactor(1, 70)

# 随机生成数据
x = np.arange(窗口大小)
x2 = np.arange(频谱图显示大小)
y1 = np.random.normal(size=窗口大小)
y2 = np.sin(np.linspace(0, 20, 频谱图显示大小))
y3 = np.random.normal(size=窗口大小)
y4 = np.cos(np.linspace(0, 20, 频谱图显示大小))
y5 = np.random.normal(size=窗口大小)
y6 = np.sin(np.linspace(0, 20, 频谱图显示大小))
y7 = np.random.normal(size=窗口大小)
y8 = np.cos(np.linspace(0, 20, 频谱图显示大小))

# 第一行（4个子图）
p1 = waveform_widget.addPlot(title="Plot 1")
curve1 = p1.plot(x, y1, pen='r')
p2 = waveform_widget.addPlot(title="Plot 2")
curve2 = p2.plot(x2, y2, pen='r')
# p2.setYRange(0, 0.000001)

waveform_widget.nextRow()
p3 = waveform_widget.addPlot(title="Plot 3")
curve3 = p3.plot(x, y3, pen='g')
p4 = waveform_widget.addPlot(title="Plot 4")
curve4 = p4.plot(x2, y4, pen='g')
# p4.setYRange(0, 0.000001)

waveform_widget.nextRow()
p5 = waveform_widget.addPlot(title="Plot 5")
curve5 = p5.plot(x, y5, pen='c')
p6 = waveform_widget.addPlot(title="Plot 6")
curve6 = p6.plot(x2, y6, pen='c')

waveform_widget.nextRow()
p7 = waveform_widget.addPlot(title="Plot 7")
curve7 = p7.plot(x, y7, pen='orange')
p8 = waveform_widget.addPlot(title="Plot 8")
curve8 = p8.plot(x2, y8, pen='orange')


def plot_welch_psd(signal, fs=500):
    """
    计算并绘制Welch功率谱密度（PSD）

    参数:
    - signal: 输入的信号 (长度为500的np数组)
    - fs: 采样频率，默认为500Hz

    返回:
    - freqs: 频率数组
    - psd: 功率谱密度数组
    """
    freqs, psd = welch(signal, fs=fs, nperseg=500)
    return freqs, psd


活动窗口数据列表 = [np.zeros(窗口大小 + (每次处理大小*15*2), dtype=np.uint16) for _ in range(通道数)]
录制数据列表 = [np.zeros(1, dtype=np.uint16) for _ in range(通道数)]
录制参数 = 0


# 更新波形图
def 更新波形图(data):
    global values
    global 活动窗口数据列表
    global 录制参数

    for channel in range(通道数):
        start_index = channel * 2
        value = (data[start_index] << 8) | data[start_index + 1]
        value = value / 65535 * 2.5 - 1.25
        values[channel].append(value)

    if all(len(row) == 每次处理大小 for row in values):
        for channel in range(通道数):
            信号数组 = np.array(values[channel])

            if 录制参数 == 1:
                录制数据列表[channel] = np.append(录制数据列表[channel], 信号数组)

            活动窗口数据列表[channel] = 活动窗口数据列表[channel][每次处理大小:]
            信号数组 = np.append(活动窗口数据列表[channel], 信号数组)
            活动窗口数据列表[channel] = 信号数组
            # 滤波后信号 = 信号数组
            滤波后信号 = 滤波器.带通滤波器(信号数组)
            滤波后信号 = 滤波器.带阻滤波器(滤波后信号)
            滤波后信号 = 滤波器.带阻滤波器2(滤波后信号)
            滤波后信号 = 滤波后信号[每次处理大小*15:len(滤波后信号) - 每次处理大小*15]
            _, 频谱图数据 = plot_welch_psd(滤波后信号[-500:])

            eval(f'curve{(channel + 1) * 2 - 1}').setData(x, 滤波后信号)
            eval(f'curve{(channel + 1) * 2}').setData(x2, 频谱图数据[:频谱图显示大小])

            values[channel] = []


检测值 = 1
数字序列 = 0


# 接收数据
def 接收数据():
    while True:
        if 检测值 == 1:
            data, _ = sock.recvfrom(8)
            更新波形图(data)


# 启动接收数据的线程
import threading


def 启动函数():
    global 检测值
    log_text_edit.clear()
    检测值 = 1
    print("数据接收启动")


def 暂停函数():
    global 检测值
    log_text_edit.clear()
    检测值 = 0
    print("数据接收暂停")


def 清空日志函数():
    log_text_edit.clear()


def 录制函数():
    btn3.setEnabled(False)
    btn3.setStyleSheet("color: gray; background-color: lightgray;")

    global 录制参数
    global 数字序列
    log_text_edit.clear()
    print("开始录制，请你想象下面数字序列")
    QCoreApplication.processEvents()  # 强制刷新界面，显示输出
    time.sleep(1)
    for 秒数 in range(4):
        log_text_edit.clear()
        print("还剩" + str(4 - (秒数 + 1)) + "s开始")
        QCoreApplication.processEvents()  # 强制刷新界面，显示输出
        time.sleep(0.5)
    log_text_edit.clear()
    数字序列 = str(滤波器.随机生成数字序列())
    数字序列_clean = 数字序列.strip('[]')
    动作列表 = ["双手前抬", "双手后抬", "左手左台", "右手右台"]

    print("请你想象下面动作: ", 动作列表[int(数字序列_clean)-1])
    btn4.setEnabled(True)
    btn4.setStyleSheet("")
    录制参数 = 1
    QCoreApplication.processEvents()  # 强制刷新界面，显示输出


def 结束录制函数():
    global 数字序列
    global 录制数据列表
    global 录制参数

    文件列表 = sorted(os.listdir("训练数据"))

    while True:
        if 滤波器.检查数组长度一致性(录制数据列表):
            log_text_edit.clear()

            for 信号 in range(通道数):
                滤波后信号 = 滤波器.带通滤波器(录制数据列表[信号])
                滤波后信号 = 滤波器.带阻滤波器(滤波后信号)
                录制数据列表[信号] = 滤波器.带阻滤波器2(滤波后信号)

            print("开始保存")
            录制参数 = 0
            滤波器.处理采样信号(录制数据列表, [数字序列])
            print("保存成功!")
            print("现有数据数：" + str(len(文件列表)))
            btn3.setEnabled(True)
            btn3.setStyleSheet("")
            btn4.setEnabled(False)
            btn4.setStyleSheet("color: gray; background-color: lightgray;")
            录制数据列表 = [np.zeros(1, dtype=np.uint16) for _ in range(通道数)]
            return 0


def 取消保存():
    global 录制数据列表
    log_text_edit.clear()
    btn3.setEnabled(True)
    btn3.setStyleSheet("")
    btn4.setEnabled(False)
    btn4.setStyleSheet("color: gray; background-color: lightgray;")
    录制数据列表 = [np.zeros(1, dtype=np.uint16) for _ in range(通道数)]


btn1.clicked.connect(启动函数)
btn2.clicked.connect(暂停函数)
btn3.clicked.connect(录制函数)
btn4.clicked.connect(结束录制函数)
btn5.clicked.connect(清空日志函数)
btn6.clicked.connect(取消保存)

recv_thread = threading.Thread(target=接收数据, daemon=True)
recv_thread.start()


# 将标准输出重定向到文本框
class StreamToTextEdit:
    def __init__(self, text_edit):
        self.text_edit = text_edit

    def write(self, text):
        self.text_edit.append(text)

    def flush(self):
        pass


sys.stdout = StreamToTextEdit(log_text_edit)

if __name__ == '__main__':
    main_window.show()
    QtWidgets.QApplication.instance().exec()
