#请注意：这个程序计算操作左峰共振时，在实验室坐标系中的演化
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import mplcursors

#参数设定(有磁场),不妨左峰跃迁频率1000MHz、右峰1700MHz
Delta = 2 * np.pi *  700 * 10 ** (-3) # Zeeman劈裂不妨700MMHz
Rabi_X = 2 * np.pi * 0.5 * 5 * 10 ** (-3) # X方向微波拉比频率不妨5MHz

H_MWX = Qobj([[0, 0, 0],[0, 0, Rabi_X],[0, Rabi_X, 0]]) #基为{+1,0,-1} 
H_Level = Qobj([[Delta,0,0],[0,0,0],[0,0,0]])
H_T =  H_MWX + H_Level# 在左峰旋转坐标系下的总哈密顿量 = 微波哈密顿量 + 能级哈密顿量

f_MW = 2 * np.pi * 1000 * 10 ** (-3)
Rabi_Z = 2 * np.pi * 0.5 * 5 * 10 ** (-3) # 考虑c-axis垂直于切面的样品，辐射结构微波场在PL6的Z方向微波拉比频率不可忽略，不妨5MHz
H_MWZ = Qobj([[1, 0, 0],[0, 0, 0],[0, 0, -1]]) #基为{+1,0,-1}    
def drive_coeff(t, args):
    """
    f(t) =  Rabi_Z * sin(f_MW * t)
    args 中若有需要的参数可以提前放入
    """
    return args["Rabi_Z"] * np.sin(args["f_MW"] * t)

# H 列表的写法：
#  - 第一个元素是一个与时间无关的项 H0
#  - 之后可以添加若干 [H_i, func_i] 这样的列表，表示含时项
H = [H_T, [H_MWZ, drive_coeff]]

# 定义初始状态（假设实验操控+1和0能级）
psi0 = Qobj([[0],[1],[0]])   # 初态为0

# 定义时间点
tlist = np.linspace(0, 10000, 10000) 

# ========== 4. 调用求解器进行演化 ==========
# 注意，我们把需要的参数 A, omega 放入 args
args = {"f_MW": f_MW, "Rabi_Z": Rabi_Z}
# 若没有耗散，可以用 sesolve
result = sesolve(H, psi0, tlist, e_ops=[], args=args)

# ========== 5. 处理并可视化结果 ==========
Popu_0 = expect(Qobj([[0, 0, 0],[0, 1, 0],[0, 0, 0]]), result.states)
Popu_p = expect(Qobj([[1, 0, 0],[0, 0, 0],[0, 0, 0]]), result.states)
Popu_m = expect(Qobj([[0, 0, 0],[0, 0, 0],[0, 0, 1]]), result.states)

# 绘制0,+,-能级占据概率随时间的变化
N = int(len(tlist)/10)
plt.figure(figsize=(8, 5))
plt.plot(tlist[:N], Popu_0[:N], label=f"0")
plt.plot(tlist[:N], Popu_p[:N], label=f"+1")
plt.plot(tlist[:N], Popu_m[:N], label=f"-1")
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
cursor = mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f'({sel.target[0]}, {sel.target[1]})'))
plt.show()