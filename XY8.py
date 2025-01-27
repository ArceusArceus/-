"""
=============================================================================
**量子比特在 O-U 噪声 + XY8 动力学去耦 序列下的数值模拟 (并行版,修正后)**
- 使用 QuTiP 的 parallel_map 并行扫描多个 T
- 避免 "Can't get attribute ..." 错误: single_T_job 在脚本顶层
- 将其余不变参数用 functools.partial 绑定, 只让 T_val 为可变

包含:
1) O-U (Ornstein-Uhlenbeck) 纯退相干噪声
2) XY8 序列 (8个脉冲)
3) 对不同 T 并行执行蒙特卡洛 (N_runs 次噪声采样)
4) 用 e^{-(T/T2)^3} 做非线性拟合, 并对比理论 T2
5) 画出 ln(-ln(C)) vs ln(T) 图, 检验斜率是否约等于 3
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip.parallel import parallel_map
from scipy.optimize import curve_fit
from functools import partial  # 用于绑定参数给 single_T_job

##############################################################################
# (0) 理论 T2 计算公式
##############################################################################
def calc_theoretical_T2(N_pulse, tau_c, sigma):
    """
    根据理论分析:
    T2 = (N_pulse)^(2/3) * (12*tau_c / sigma^2)^(1/3)
    """
    return (N_pulse)**(2./3.) * (12.*tau_c / (sigma**2))**(1./3.) / np.sqrt(2)

##############################################################################
# (A) 生成 O-U (Ornstein-Uhlenbeck) 噪声的函数
##############################################################################
def generate_ou_noise(t_list, tau_c, sigma, B0=0.0, seed=None):
    """
    生成离散的 Ornstein-Uhlenbeck (O-U) 噪声序列:
      dB = - (1/tau_c)*B dt + sigma * dW
    """
    if seed is not None:
        np.random.seed(seed)
    N = len(t_list)
    B_arr = np.zeros(N)
    B_arr[0] = B0
    dt_list = np.diff(t_list)

    for i in range(1, N):
        dt = dt_list[i-1]
        dW = np.random.normal(0, np.sqrt(dt))
        B_arr[i] = ( B_arr[i-1]
                     - (dt/tau_c)*B_arr[i-1]
                     + sigma*np.sqrt(2.0/tau_c)*dW )
    return B_arr

##############################################################################
# (B) 计算 XY8 序列中 8 个脉冲的中心时刻 (可选用于标线)
##############################################################################
def compute_xy8_pulse_centers_advanced(T, t_pi_x, t_pi_y, pulses):
    """
    根据与 run_single_experiment_xy8_advanced 相同的时间分段规则,
    计算 8 个脉冲的中心时刻, 以方便在绘图中标注.
    """
    N_x = sum([p=='X' for p in pulses])
    N_y = sum([p=='Y' for p in pulses])
    total_pulse_time = N_x*t_pi_x + N_y*t_pi_y
    T_free = T - total_pulse_time
    if T_free < 0:
        raise ValueError(f"总脉冲时长 {total_pulse_time} 超过 T={T}")
    if len(pulses)!=8:
        raise ValueError("此函数仅实现 8 脉冲分段的 XY8.")

    first_int  = T_free/16
    last_int   = T_free/16
    middle_int = T_free/8
    intervals  = [first_int] + [middle_int]*7 + [last_int]

    def get_pulse_length(ptype):
        if ptype=='X':
            return t_pi_x
        elif ptype=='Y':
            return t_pi_y
        else:
            raise ValueError("pulses must be 'X' or 'Y'.")

    centers = []
    t_accum = 0.0
    for i in range(8):
        t_accum += intervals[i]
        t_pulse = get_pulse_length(pulses[i])
        center = t_accum + t_pulse/2
        centers.append(center)
        t_accum += t_pulse
    return centers

##############################################################################
# (C) 单次实验: 执行 XY8 动力学去耦
##############################################################################
def run_single_experiment_xy8_advanced(
    T,
    t_pi_x,
    t_pi_y,
    epsilon_x,
    epsilon_y,
    ny, nz,
    mx, mz,
    pulses,
    psi0,
    omega0,
    ou_tlist,
    ou_noise,
    N_free_step,
    N_pulse_step
):
    """
    在总时长 T 内插入 XY8 序列(8脉冲). 
    区分 X/Y 脉冲的时长(t_pi_x / t_pi_y), 幅度误差(epsilon_x / epsilon_y), 
    以及转轴误差(由ny,nz / mx,mz指定).
    同时考虑给定的 O-U 噪声轨迹 ou_noise(与 ou_tlist 对应).
    """
    N_x = sum([p=='X' for p in pulses])
    N_y = sum([p=='Y' for p in pulses])
    total_pulse_time = N_x*t_pi_x + N_y*t_pi_y
    T_free = T - total_pulse_time
    if T_free < 0:
        raise ValueError(f"总脉冲时长 {total_pulse_time} > T={T}")
    if len(pulses)!=8:
        raise ValueError("仅支持8脉冲分段的XY8逻辑.")

    first_int  = T_free/16
    last_int   = T_free/16
    middle_int = T_free/8
    intervals  = [first_int] + [middle_int]*7 + [last_int]

    sx, sy, sz = sigmax(), sigmay(), sigmaz()
    H0 = (omega0/2)*sz

    def beta_t(t, args=None):
        if t<0:
            return 0.0
        if t >= ou_tlist[-1]:
            return ou_noise[-1]
        idx = np.searchsorted(ou_tlist, t)
        if idx==0:
            return ou_noise[0]
        if idx>=len(ou_tlist):
            return ou_noise[-1]
        t1 = ou_tlist[idx-1]
        t2 = ou_tlist[idx]
        f1 = ou_noise[idx-1]
        f2 = ou_noise[idx]
        return f1 + (f2 - f1)*((t - t1)/(t2 - t1))

    H_noise = [sz, beta_t]

    axis_x_x = np.sqrt(max(0.0, 1.0 - ny**2 - nz**2))
    axis_x_y = ny
    axis_x_z = nz
    axis_y_x = mx
    axis_y_y = np.sqrt(max(0.0, 1.0 - mx**2 - mz**2))
    axis_y_z = mz

    sx_op = axis_x_x*sx + axis_x_y*sy + axis_x_z*sz
    sy_op = axis_y_x*sx + axis_y_y*sy + axis_y_z*sz

    def get_pulse_param(ptype):
        if ptype=='X':
            return t_pi_x, epsilon_x, sx_op
        elif ptype=='Y':
            return t_pi_y, epsilon_y, sy_op
        else:
            raise ValueError("Pulse must be 'X' or 'Y'")

    current_state = psi0
    all_times = [0.0]
    all_states = [psi0]
    t_accum = 0.0

    for i in range(8):
        dt_free = intervals[i]
        if dt_free>1e-12:
            t_start = t_accum
            t_end = t_accum + dt_free
            t_span = np.linspace(t_start, t_end, N_free_step)
            H_free = [H0, H_noise]

            res_free = mesolve(H_free, current_state, t_span, [], [],
                               options=Options(store_final_state=True))
            current_state = res_free.final_state
            t_accum = t_end
            all_times.extend(t_span[1:])
            all_states.extend(res_free.states[1:])

        pt = pulses[i]
        t_pulse, eps_pulse, op_pulse = get_pulse_param(pt)
        amp = (np.pi + eps_pulse)/(2.0 * t_pulse)

        H_pulse = [H0, H_noise, [op_pulse, lambda t,args: amp]]
        t_start = t_accum
        t_end = t_accum + t_pulse
        t_span = np.linspace(t_start, t_end, N_pulse_step)

        res_pulse = mesolve(H_pulse, current_state, t_span, [], [],
                            options=Options(store_final_state=True))
        current_state = res_pulse.final_state
        t_accum = t_end
        all_times.extend(t_span[1:])
        all_states.extend(res_pulse.states[1:])

    dt_free = intervals[8]
    if dt_free>1e-12:
        t_start = t_accum
        t_end = t_accum + dt_free
        t_span = np.linspace(t_start, t_end, N_free_step)
        H_free = [H0, H_noise]
        res_free = mesolve(H_free, current_state, t_span, [], [],
                           options=Options(store_final_state=True))
        current_state = res_free.final_state
        t_accum = t_end
        all_times.extend(t_span[1:])
        all_states.extend(res_free.states[1:])

    return np.array(all_times), all_states

##############################################################################
# (D) 蒙特卡洛循环: 多次生成 O-U 噪声, 对全程结果做平均 (Bloch分量)
##############################################################################
def run_monte_carlo_xy8_advanced(
    N_runs,
    T,
    t_pi_x, t_pi_y,
    epsilon_x, epsilon_y,
    ny, nz, mx, mz,
    tau_c, sigma, B0,
    pulses,
    psi0,
    omega0,
    N_free_step,
    N_pulse_step
):
    """
    对单个 T 做 N_runs 次 O-U 噪声采样, 并计算 Bloch 分量平均.
    """
    N_x = sum([p=='X' for p in pulses])
    N_y = sum([p=='Y' for p in pulses])
    total_pulse_time = N_x*t_pi_x + N_y*t_pi_y
    if T < total_pulse_time:
        raise ValueError(f"T={T} 小于脉冲总时长={total_pulse_time}")

    N_noise_grid = 2000
    ou_tlist = np.linspace(0, T, N_noise_grid)

    ou_noise_dummy = generate_ou_noise(ou_tlist, tau_c, sigma, B0, seed=0)
    test_times, test_states = run_single_experiment_xy8_advanced(
        T,
        t_pi_x, t_pi_y,
        epsilon_x, epsilon_y,
        ny, nz,
        mx, mz,
        pulses,
        psi0,
        omega0,
        ou_tlist,
        ou_noise_dummy,
        N_free_step,
        N_pulse_step
    )
    common_tlist = test_times
    N_points = len(common_tlist)

    sx, sy, sz = sigmax(), sigmay(), sigmaz()
    bloch_x_sum = np.zeros(N_points)
    bloch_y_sum = np.zeros(N_points)
    bloch_z_sum = np.zeros(N_points)

    for _ in range(N_runs):
        ou_noise = generate_ou_noise(ou_tlist, tau_c, sigma, B0)
        all_times, all_states = run_single_experiment_xy8_advanced(
            T,
            t_pi_x, t_pi_y,
            epsilon_x, epsilon_y,
            ny, nz,
            mx, mz,
            pulses,
            psi0,
            omega0,
            ou_tlist,
            ou_noise,
            N_free_step,
            N_pulse_step
        )
        for i in range(N_points):
            st = all_states[i]
            bloch_x_sum[i] += expect(sx, st)
            bloch_y_sum[i] += expect(sy, st)
            bloch_z_sum[i] += expect(sz, st)

    bx_mean = bloch_x_sum / N_runs
    by_mean = bloch_y_sum / N_runs
    bz_mean = bloch_z_sum / N_runs

    return common_tlist, bx_mean, by_mean, bz_mean

##############################################################################
# (E.1) 并行执行的任务函数: single_T_job
##############################################################################
def single_T_job(
    T_val,            # 可变的参数(本例只变 T)
    N_runs,
    t_pi_x, t_pi_y,
    epsilon_x, epsilon_y,
    ny, nz, mx, mz,
    tau_c, sigma, B0,
    pulses,
    psi0,
    omega0,
    N_free_step,
    N_pulse_step
):
    """
    用于 parallel_map: 对给定 T_val, 执行 run_monte_carlo_xy8_advanced 并返回末态 <X>.
    注意: 该函数**必须**定义在脚本顶层(非 if __name__ 内), 
    否则 Windows 下子进程无法序列化函数对象.

    返回:
      末态 <X> (coherence).
    """
    times, bx_mean, by_mean, bz_mean = run_monte_carlo_xy8_advanced(
        N_runs,
        T_val,
        t_pi_x, t_pi_y,
        epsilon_x, epsilon_y,
        ny, nz, mx, mz,
        tau_c, sigma, B0,
        pulses,
        psi0,
        omega0,
        N_free_step,
        N_pulse_step
    )
    return bx_mean[-1]

##############################################################################
# (E.2) main: 并行扫描 T, 拟合, 画图
##############################################################################
if __name__ == "__main__":
    #------------------------------
    # 1) 在 main 中统一设置参数
    #------------------------------
    tau_c   = 25
    sigma   = 3.6
    B0      = 0.0

    t_pi_x  = 0.008
    t_pi_y  = 0.008
    epsilon_x = 0.0
    epsilon_y = 0.0
    ny, nz  = 0.0, 0.0
    mx, mz  = 0.0, 0.0

    pulses_xy8 = ['X','Y','X','Y','Y','X','Y','X']
    psi0 = (basis(2,0)+basis(2,1)).unit()
    omega0 = 0.0

    N_runs       = 100
    N_free_step  = 100
    N_pulse_step = 50

    # 定义 T 扫描区间
    N_x = sum([p=='X' for p in pulses_xy8])
    N_y = sum([p=='Y' for p in pulses_xy8])
    total_pulse_time = N_x*t_pi_x + N_y*t_pi_y
    T_min = total_pulse_time
    T_max = 30.0
    NT    = 31
    T_values = np.linspace(T_min, T_max, NT)

    # 理论 T2
    N_pulse = len(pulses_xy8)
    T2_theory = calc_theoretical_T2(N_pulse, tau_c, sigma)
    print(f"Theoretical T2 = {T2_theory:.3f}  "
          f"(N_pulse={N_pulse}, tau_c={tau_c}, sigma={sigma})")

    #------------------------------
    # 2) 用 partial 固定不变参数, 并行执行 single_T_job(T_val)
    #------------------------------
    # 准备 partial 函数, 绑定所有固定参数, 只留 T_val 作可变
    job_func = partial(
        single_T_job,
        N_runs=N_runs,
        t_pi_x=t_pi_x,
        t_pi_y=t_pi_y,
        epsilon_x=epsilon_x,
        epsilon_y=epsilon_y,
        ny=ny, nz=nz,
        mx=mx, mz=mz,
        tau_c=tau_c,
        sigma=sigma,
        B0=B0,
        pulses=pulses_xy8,
        psi0=psi0,
        omega0=omega0,
        N_free_step=N_free_step,
        N_pulse_step=N_pulse_step
    )

    # 并行执行
    final_coherence_list = parallel_map(job_func, T_values)
    final_coherence_list = np.array(final_coherence_list)

    #------------------------------
    # 3) 拟合 C(T)=exp(-(T/T2)^3)
    #------------------------------
    def model_C(T, T2):
        return np.exp(- (T / T2)**3)

    p0 = [1.0]
    popt, pcov = curve_fit(model_C, T_values, final_coherence_list, p0=p0)
    T2_fit = popt[0]
    T2_fit_err = np.sqrt(np.diag(pcov))[0]
    print(f"Fit result: T2_fit = {T2_fit:.3f} ± {T2_fit_err:.3f}")

    #------------------------------
    # 4) 画图1: C vs T + 拟合 + 理论T2
    #------------------------------
    T_dense = np.linspace(T_min, T_max, 200)
    C_fit   = model_C(T_dense, T2_fit)

    plt.figure(figsize=(7,5))
    plt.plot(T_values, final_coherence_list, 'o', label='Data')
    label_fit = (f"Fit T2={T2_fit:.3f}±{T2_fit_err:.3f}\nTheory T2={T2_theory:.3f}")
    plt.plot(T_dense, C_fit, 'r--', label=label_fit)

    plt.xlabel("Total Time T")
    plt.ylabel("Final Coherence <X>")
    plt.title("Coherence vs T (XY8 under O-U noise) - parallel_map")
    plt.grid(True)
    plt.legend()
    plt.show()

    #------------------------------
    # 5) 画图2: ln(-ln(C)) vs ln(T)
    #------------------------------
    T_vals_plot = []
    Y_vals_plot = []
    for (Tv, Cf) in zip(T_values, final_coherence_list):
        if 0 < Cf < 1:
            val = -np.log(Cf)
            if val>0:
                T_vals_plot.append(Tv)
                Y_vals_plot.append(np.log(val))

    T_vals_plot = np.array(T_vals_plot)
    Y_vals_plot = np.array(Y_vals_plot)

    plt.figure(figsize=(6,5))
    xlog = np.log(T_vals_plot)
    ylog = Y_vals_plot
    plt.plot(xlog, ylog, 'o', label='Data')

    def linear_func(x,a,b):
        return a*x + b
    p_lin, pcov_lin = curve_fit(linear_func, xlog, ylog, p0=[3,0])
    a_lin, b_lin = p_lin
    da_lin, db_lin = np.sqrt(np.diag(pcov_lin))

    xfit = np.linspace(min(xlog), max(xlog), 100)
    yfit = a_lin*xfit + b_lin
    plt.plot(xfit, yfit, 'r--', label=f"Linear slope={a_lin:.2f}±{da_lin:.2f}")

    plt.xlabel("ln(T)")
    plt.ylabel("ln(-ln(C))")
    plt.title("Check slope ~ 3 for e^{-(T/T2)^3}")
    plt.grid(True)
    plt.legend()
    plt.show()

    print("In ln(-ln(C)) vs ln(T):")
    print(f"  slope = {a_lin:.3f} ± {da_lin:.3f}, intercept={b_lin:.3f}±{db_lin:.3f}")
    print("  (If slope ~ 3 => data ~ e^{-(T/T2)^3}).")
