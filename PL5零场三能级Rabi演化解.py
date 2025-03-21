#请注意：这个程序计算操作左峰共振时，在左峰的旋转坐标系中的演化
#旋转变换V = [[e^-iwt 0 0 ],[0 1 0],[0 0 e^-iwt]] w=D-E
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import mplcursors

#参数设定（零场）
Delta = 2 * np.pi *  30 * 10 ** (-3) # 劈裂30MHz
Rabi_R = 2 * np.pi * 0.5 * 5 * 10 ** (-3) # 右峰操控强度10MHz
Rabi_L = 2 * np.pi * 0.5 * 3 * 10 ** (-3) # 左峰操控强度3MHz

H_MW = Qobj([[0, Rabi_R, 0],[Rabi_R, 0, 1j * Rabi_L],[0, -1j * Rabi_L, 0]]) #基为{+,0,-}
H_Level = Qobj([[Delta,0,0],[0,0,0],[0,0,0]])
H_T =  H_MW + H_Level# 在左峰旋转坐标系下的总哈密顿量 = 微波哈密顿量 + 能级哈密顿量

# 定义初始状态（假设实验操控+1和0能级）
psi0 = Qobj([[0],[1],[0]])   # 初态为0

# 定义时间点
tlist = np.linspace(0, 10000, 10000) 

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

    

# 绘制0,+,-能级占据概率随时间的变化
N = int(len(tlist)/10)
plt.figure(figsize=(8, 5))
plt.plot(tlist[:N], Popu_0[:N], label=f"0")
plt.plot(tlist[:N], Popu_p[:N], label=f"+")
plt.plot(tlist[:N], Popu_m[:N], label=f"-")
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f'({sel.target[0]}, {sel.target[1]})'))
plt.show()

# 用cos拟合0能级曲线，观察拟合出的Rabi频率与实际设定的左峰频率的差别
Popu_0 = Popu_0 - np.mean(Popu_0)
Spectrum = np.fft.fft(Popu_0)
Spectrum = np.abs(Spectrum)
n = len(tlist)  # 数据点数
T = tlist[1] - tlist[0]  # 采样间隔
freq = np.fft.fftfreq(n, T)# 频率轴
plt.figure()
plt.plot(freq, Spectrum)
plt.title('FFT of the signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f'({sel.target[0]}, {sel.target[1]})'))
plt.show()
print('Rabi Freq = ',freq[np.argmax(Spectrum)])

# plot含时Rabi (Pulse-CW)
t_Pi = 0.5 * abs(1/freq[np.argmax(Spectrum)])
Popu_0_f = []

delta_list = np.arange(-50,100,0.01) #干脆把右峰的Pulse CW也plot出来
delta_list.reshape(-1,1)
delta_list_H = 2 * np.pi * delta_list * 10 ** (-3)
for delta in delta_list_H:
    H_Level = Qobj([[Delta-delta,0,0],[0,0,0],[0,0,-delta]])
    H_T =  H_MW + H_Level
    O_Evolve = (-1j * t_Pi * H_T).expm()
    psi_f = O_Evolve * psi0
    Coeff_0_f = psi_f[1]
    Popu_0_f.append(abs(Coeff_0_f[0,0]) ** 2)

plt.figure(figsize=(8, 5))
plt.plot(delta_list, Popu_0_f, label=f"0")
plt.xlabel('delta from Left Peak Resonance, positive = higher freq than Left Peak')
plt.ylabel('Population')
plt.legend()
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f'({sel.target[0]}, {sel.target[1]})'))
plt.show()

# 换到模拟的含时Rabi的最低点，注意此时频率不在左峰，比左峰略高
N = int(55/0.01)
Popu_0_f = Popu_0_f[:N] #截取左峰，不要右峰
delta = delta_list_H[np.argmin(Popu_0_f)]
print('delta = ',delta/2/np.pi* 10 ** 3)
H_Level = Qobj([[Delta-delta,0,0],[0,0,0],[0,0,-delta]])
H_T =  H_MW + H_Level

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

N = int(len(tlist)/10)
plt.figure(figsize=(8, 5))
plt.plot(tlist[:N], Popu_0[:N], label=f"0")
plt.plot(tlist[:N], Popu_p[:N], label=f"+")
plt.plot(tlist[:N], Popu_m[:N], label=f"-")
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f'({sel.target[0]}, {sel.target[1]})'))
plt.show()
Popu_0 = Popu_0 - np.mean(Popu_0)
Spectrum = np.fft.fft(Popu_0)
Spectrum = np.abs(Spectrum)
n = len(tlist)  # 数据点数
T = tlist[1] - tlist[0]  # 采样间隔
freq = np.fft.fftfreq(n, T)# 频率轴
plt.figure()
plt.plot(freq, Spectrum)
plt.title('FFT of the signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f'({sel.target[0]}, {sel.target[1]})'))
plt.show()
print('Rabi Freq = ',freq[np.argmax(Spectrum)])




