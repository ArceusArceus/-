#PL5 零场 右峰 XY-8
#旋转变换V = [[e^-iwt 0 0 ],[0 1 0],[0 0 e^-iwt]] w约为D+E
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import mplcursors

#参数设定
Delta = 30
Rabi_R = 3
Rabi_L = 10
T_Pi = 0.95 * 0.5 * 1 / Rabi_R * 10 ** 3 # 右峰操控Pi脉冲长度
Delta = 2 * np.pi * Delta  * 10 ** (-3) # 劈裂30MHz
Rabi_R = 2 * np.pi * 0.5 * Rabi_R * 10 ** (-3) # 右峰操控强度10MHz
Rabi_L = 2 * np.pi * 0.5 * Rabi_L * 10 ** (-3) # 左峰操控强度3MHz


H_MW_0 = Qobj([[0, Rabi_R, 0],[Rabi_R, 0, 1j * Rabi_L],[0, -1j * Rabi_L, 0]]) #基为{+,0,-}
H_MW_90 = Qobj([[0, -1j * Rabi_R, 0],[1j * Rabi_R, 0, 1j * 1j * Rabi_L],[0, -1j * -1j * Rabi_L, 0]]) #基为{+,0,-}
H_MW_180 = Qobj([[0, -Rabi_R, 0],[-Rabi_R, 0, -1 * 1j * Rabi_L],[0, -1 * -1j * Rabi_L, 0]]) #基为{+,0,-}
H_MW_270 = Qobj([[0, 1j * Rabi_R, 0],[-1j * Rabi_R, 0, -1j * 1j * Rabi_L],[0, 1j * -1j * Rabi_L, 0]]) #基为{+,0,-}


delta = 2 * np.pi * 0 * 10 ** (-3) #距右峰偏共振
H_Level = Qobj([[-delta,0,0],[0,0,0],[0,0,-Delta-delta]])

H_T_0 =   H_MW_0 + H_Level# 在右峰旋转坐标系下操控的总哈密顿量 = 微波哈密顿量 + 能级哈密顿量
H_T_90 =   H_MW_90 + H_Level
H_T_180 =   H_MW_180 + H_Level
H_T_270 =   H_MW_270 + H_Level

O_halfPi_0 = (-1j * T_Pi / 2 * H_T_0).expm()
O_halfPi_90 = (-1j * T_Pi / 2 * H_T_90).expm()
O_Pi_0 = (-1j * T_Pi * H_T_0).expm()
O_Pi_90 = (-1j * T_Pi * H_T_90).expm()
O_Pi_180 = (-1j * T_Pi * H_T_180).expm()
O_Pi_270 = (-1j * T_Pi * H_T_270).expm()


# 定义初始状态（假设实验操控+1和0能级）                                             
psi0 = Qobj([[0],[1],[0]])   # 初态为0

# 扫描Pi脉冲间隔
taulist = np.linspace(0, 8000, 1000) 


## T2情形
Popu_halfpiRead = [] #随tau变化的序列最终0态布居（因为默认+-对比度相同，最中对比度即取决于0态布居）
Popu_3halfpiRead = []

for tau in taulist:
    O_half_tau = (-1j * tau / 2 * H_Level).expm()
    psi1 = O_halfPi_0 * psi0 
    psi2 = O_half_tau * psi1 
    psi3 = O_Pi_90 * psi2 
    psi4 = O_half_tau * psi3
    psi5 = O_halfPi_0 * psi4
    Coeff = psi5[1]
    Popu_halfpiRead.append(abs(Coeff[0,0]) ** 2)
    psi6 = O_halfPi_0 * O_halfPi_0 * psi5
    Coeff = psi6[1]
    Popu_3halfpiRead.append(abs(Coeff[0,0]) ** 2)
Popu_Substract = np.subtract(Popu_3halfpiRead,Popu_halfpiRead)
# 绘制T2曲线
plt.figure(figsize=(8, 5))
plt.plot(np.array(taulist) * (10 ** (-3)), Popu_halfpiRead, label=f"pi/2 read")
plt.plot(np.array(taulist) * (10 ** (-3)), Popu_3halfpiRead, label=f"3pi/2 read")
plt.plot(np.array(taulist) * (10 ** (-3)), Popu_Substract, label=f"Substract")
plt.xlabel('Tau')
plt.ylabel('Population in 0')
plt.legend()
plt.title('T2 Curve')
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f'({sel.target[0]}, {sel.target[1]})'))
plt.show()
# 加入e指数衰减，不妨T2 = 10u
T2 = 4.41718 * 10 ** 3
Decay_Rate = 1 / T2
Decay_coeff = np.exp(- Decay_Rate * taulist)
plt.figure(figsize=(8, 5))
plt.plot(np.array(taulist) * (10 ** (-3)), np.multiply(Decay_coeff,Popu_Substract), label=f"Substract with 4.41718u T2 Decay")
plt.xlabel('Tau')
plt.ylabel('Population in 0')
plt.legend()
plt.title('T2 Curve')
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f'({sel.target[0]}, {sel.target[1]})'))
plt.show()


## CPMG-8情形
Popu_halfpiRead = [] #随tau变化的序列最终0态布居（因为默认+-对比度相同，最中对比度即取决于0态布居）
Popu_3halfpiRead = []

for tau in taulist:
    O_half_tau = (-1j * tau / 2 * H_Level).expm()
    psi1 = O_halfPi_0 * psi0 
    psi2 = O_half_tau * psi1 
    psi3 = O_Pi_0 * psi2 
    psi4 = O_half_tau * O_half_tau * psi3 
    psi5 = O_Pi_0 * psi4 
    psi6 = O_half_tau * O_half_tau * psi5 
    psi7 = O_Pi_0 * psi6 
    psi8 = O_half_tau * O_half_tau * psi7
    psi9 = O_Pi_0 * psi8 
    psi10 = O_half_tau * O_half_tau * psi9
    psi11 = O_Pi_0 * psi10
    psi12 = O_half_tau * O_half_tau * psi11
    psi13 = O_Pi_0 * psi12 
    psi14 = O_half_tau * O_half_tau * psi13
    psi15 = O_Pi_0 * psi14
    psi16 = O_half_tau * O_half_tau * psi15
    psi17 = O_Pi_0 * psi16 
    psi18 = O_half_tau * psi17
    psi19 = O_halfPi_0 * psi18
    Coeff = psi19[1]
    Popu_halfpiRead.append(abs(Coeff[0,0]) ** 2)
    psi20 = O_halfPi_0 * O_halfPi_0 * psi19
    Coeff = psi20[1]
    Popu_3halfpiRead.append(abs(Coeff[0,0]) ** 2)
Popu_Substract = np.subtract(Popu_3halfpiRead,Popu_halfpiRead)    
# 绘制CPMG-8曲线
plt.figure(figsize=(8, 5))
plt.plot(np.array(taulist) * (10 ** (-3)), Popu_halfpiRead, label=f"pi/2 read")
plt.plot(np.array(taulist) * (10 ** (-3)), Popu_3halfpiRead, label=f"3pi/2 read")
plt.plot(np.array(taulist) * (10 ** (-3)), Popu_Substract, label=f"Substract")
plt.xlabel('Tau')
plt.ylabel('Population in 0')
plt.legend()
plt.title('CPMG-8 Curve')
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f'({sel.target[0]}, {sel.target[1]})'))
plt.show()
# 加入e指数衰减，不妨T2(8) = 20u
T2 = 9.282 * 10 ** 3
Decay_Rate = 8 * 1 / T2
Decay_coeff = np.exp(- Decay_Rate * taulist)
plt.figure(figsize=(8, 5))
plt.plot(np.array(taulist) * (10 ** (-3)), np.multiply(Decay_coeff,Popu_Substract), label=f"Substract with 9.2824u T2(8) Decay")
plt.xlabel('Tau')
plt.ylabel('Population in 0')
plt.legend()
plt.title('CPMG-8 Curve')
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f'({sel.target[0]}, {sel.target[1]})'))
plt.show()


## XY-8情形
Popu_halfpiRead = [] #随tau变化的序列最终0态布居（因为默认+-对比度相同，最中对比度即取决于0态布居）
Popu_3halfpiRead = []

for tau in taulist:
    O_half_tau = (-1j * tau / 2 * H_Level).expm()
    psi1 = O_halfPi_90 * psi0 
    psi2 = O_half_tau * psi1 
    psi3 = O_Pi_0 * psi2 
    psi4 = O_half_tau * O_half_tau * psi3 
    psi5 = O_Pi_90 * psi4 
    psi6 = O_half_tau * O_half_tau * psi5 
    psi7 = O_Pi_0 * psi6 
    psi8 = O_half_tau * O_half_tau * psi7
    psi9 = O_Pi_90 * psi8 
    psi10 = O_half_tau * O_half_tau * psi9
    psi11 = O_Pi_90 * psi10
    psi12 = O_half_tau * O_half_tau * psi11
    psi13 = O_Pi_0 * psi12 
    psi14 = O_half_tau * O_half_tau * psi13
    psi15 = O_Pi_90 * psi14
    psi16 = O_half_tau * O_half_tau * psi15
    psi17 = O_Pi_0 * psi16 
    psi18 = O_half_tau * psi17
    psi19 = O_halfPi_90 * psi18
    Coeff = psi19[1]
    Popu_halfpiRead.append(abs(Coeff[0,0]) ** 2)
    psi20 = O_halfPi_90 * O_halfPi_90 * psi19
    Coeff = psi20[1]
    Popu_3halfpiRead.append(abs(Coeff[0,0]) ** 2)
Popu_Substract = np.subtract(Popu_3halfpiRead,Popu_halfpiRead)    

# 绘制XY-8曲线
plt.figure(figsize=(8, 5))
plt.plot(np.array(taulist) * (10 ** (-3)), Popu_halfpiRead, label=f"pi/2 read")
plt.plot(np.array(taulist) * (10 ** (-3)), Popu_3halfpiRead, label=f"3pi/2 read")
plt.plot(np.array(taulist) * (10 ** (-3)), Popu_Substract, label=f"Substract")
plt.xlabel('Tau')
plt.ylabel('Population in 0')
plt.legend()
plt.title('XY8 Curve')
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f'({sel.target[0]}, {sel.target[1]})'))
plt.show()
# 加入e指数衰减，不妨T2(8) = 20u
T2 = 9.282 * 10 ** 3
Decay_Rate = 8 * 1 / T2
Decay_coeff = np.exp(- Decay_Rate * taulist)
plt.figure(figsize=(8, 5))
plt.plot(np.array(taulist) * (10 ** (-3)), np.multiply(Decay_coeff,Popu_Substract), label=f"Substract with 9.2824u T2(8) Decay")
plt.xlabel('Tau')
plt.ylabel('Population in 0')
plt.legend()
plt.title('XY8 Curve')
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f'({sel.target[0]}, {sel.target[1]})'))
plt.show()