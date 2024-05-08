#请注意：这个程序计算操作左峰共振时，在左峰的旋转坐标系中的演化
#旋转变换V = [[e^-iwt 0 0 ],[0 1 0],[0 0 e^-iwt]] w=D-E
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import mplcursors

#参数设定
delta = 2 * np.pi * 0.5 * 29.0800 * 10 ** (-3) # 劈裂30MHz
Rabi_R = 2 * np.pi * 0.5 * 12.6522160260718 * 10 ** (-3) # 右峰操控强度10MHz
Rabi_L = 2 * np.pi * 0.5 * 3.40543836161306 * 10 ** (-3) # 左峰操控强度3MHz

H_MW = Qobj([[0, Rabi_R, 0],[Rabi_R, 0, 1j * Rabi_L],[0, -1j * Rabi_L, 0]]) #基为{+,0,-}
H_Level = Qobj([[delta,0,0],[0,0,0],[0,0,0]])
H_T =  H_MW + H_Level# 在左峰旋转坐标系下的总哈密顿量 = 微波哈密顿量 + 能级哈密顿量

# 定义初始状态（假设实验操控+1和0能级）
psi0 = Qobj([[0],[1],[0]])   # 初态为0

# 定义时间点
tlist = np.linspace(0, 1000, 10000) 

# 演化
Popu_0 = []
Popu_p = []
Popu_m = []
for n in range(len(tlist)):
    O_Evolve = (-1j * tlist[n] * H_T).expm()
    psi_t = O_Evolve * psi0
    Coeff_0 = psi_t[1]
    Coeff_p = psi_t[0]
    Coeff_m = psi_t[2]
    Popu_0.append(abs(Coeff_0[0,0]) ** 2)
    Popu_p.append(abs(Coeff_p[0,0]) ** 2)
    Popu_m.append(abs(Coeff_m[0,0]) ** 2)

    

# 绘制0能级占据概率随时间的变化
plt.figure(figsize=(8, 5))
plt.plot(tlist, Popu_0, label=f"0")
plt.plot(tlist, Popu_p, label=f"+")
plt.plot(tlist, Popu_m, label=f"-")
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f'({sel.target[0]}, {sel.target[1]})'))
plt.show()

# 用cos拟合0能级曲线，观察拟合出的Rabi频率与实际设定的左峰频率的差别
Popu_0 = Popu_0 - np.mean(Popu_0)
Spectrum = np.fft.fft(Popu_0)
n = len(tlist)  # 数据点数
T = tlist[1] - tlist[0]  # 采样间隔
freq = np.fft.fftfreq(n, T)# 频率轴
plt.figure()
plt.plot(freq, np.abs(Spectrum))
plt.title('FFT of the signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f'({sel.target[0]}, {sel.target[1]})'))
plt.show()
print(freq[np.argmax(Spectrum)])

