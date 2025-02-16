"""
=============================================================================
**多次重复 XY8 + O-U(Ornstein-Uhlenbeck) 噪声 并行模拟 (b为稳态std)**

主要特点:
1) generate_ou_noise: 以 b 做稳态方差 sqrt, 由 sigma = sqrt(2)*b / sqrt(tau_c),
   使用 Exact Discretization 公式生成 O-U 过程.
2) 理论 T2 = (N_pulse)^(2/3) * (12*tau_c / b^2)^(1/3) / sqrt(2), 由你指定.
3) 最终画图包含:
   - C(T) vs T + 拟合 e^{-(T/T2)^3};
   - ln(-ln(C)) vs ln(T) 作线性拟合, 检查斜率~3.

=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip.parallel import parallel_map
from scipy.optimize import curve_fit
from functools import partial

##############################################################################
# (0) 计算理论 T2
##############################################################################
def calc_theoretical_T2(N_pulse, tau_c, b):
    """
    由你指定的公式:
    T2 = (N_pulse)^(2/3) * (12*tau_c / b^2)^(1/3) / sqrt(2).
    """
    return (N_pulse**(2./3.) 
            * (12.*tau_c / (b**2))**(1./3.) )
            

##############################################################################
# (A) generate_ou_noise - Exact Discretization, sigma = sqrt(2)*b/sqrt(tau_c)
##############################################################################
def generate_ou_noise(t_list, tau_c, b, B0=None, seed=None):
    """
    在给定时间网格 t_list 上生成 1D O-U 过程 X(t):
      dX = -(1/tau_c)*X dt + sigma dW
    其中 sigma = sqrt(2)*b/sqrt(tau_c),
    => 平稳方差: X_infinity ~ Normal(0, b^2).
    
    如果 B0 为 None，则随机初始化 B0 为符合正态分布的随机数。
    """
    if seed is not None:
        np.random.seed(seed)

    # 如果B0为None，随机化 B0 为符合正态分布的随机值，均值为 0，方差为 b^2
    if B0 is None:
        B0 = np.random.normal(0, b)

    N = len(t_list)
    X_arr = np.zeros(N)
    X_arr[0] = B0

    sigma = np.sqrt(2.0) * b / np.sqrt(tau_c)  # from b => sigma

    for i in range(1, N):
        dt = t_list[i] - t_list[i-1]
        alpha = np.exp(-dt/tau_c)
        sqrt_factor = sigma * np.sqrt( (tau_c/2.)*(1. - alpha*alpha) )

        dW = np.random.normal(0, 1)
        X_arr[i] = alpha * X_arr[i-1] + sqrt_factor*dW

    return X_arr

##############################################################################
# (B) 单次实验: N脉冲序列, run_single_experiment_general
##############################################################################
def run_single_experiment_general(
    T,
    pulses,
    t_pi_x, t_pi_y,
    epsilon_x, epsilon_y,
    ny, nz, mx, mz,
    psi0,
    omega0,
    ou_tlist,
    ou_noise,
    N_free_step,
    N_pulse_step
):
    """
    对应 "多次XY8" 的通用分段逻辑:
    - pulses 有 N 个 (X/Y),
    - T_free = T - sum(t_pi_x/t_pi_y),
    - intervals = [T_free/(2N)] + [T_free/N]*(N-1) + [T_free/(2N)].
    - 在每段插入 mesolve, 构造 time-dependent H_noise(t).
    """
    N = len(pulses)
    N_x = sum([p=='X' for p in pulses])
    N_y = sum([p=='Y' for p in pulses])
    total_pulse_time = N_x*t_pi_x + N_y*t_pi_y
    T_free = T - total_pulse_time
    if T_free < 0:
        raise ValueError("T not enough to place pulses")

    intervals = [T_free/(2*N)] + [T_free/N]*(N-1) + [T_free/(2*N)]

    sx, sy, sz = sigmax(), sigmay(), sigmaz() #注意S=1/2的旋转算符是sigma/2，后面勿漏2因子
    H0 = (omega0/2)*sz #2因子

    def beta_t(t, args=None):
        if t<0:
            return 0.
        if t>=ou_tlist[-1]:
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

    H_noise = [sz/2.0, beta_t]

    # X脉冲轴
    axis_x_x = np.sqrt(max(0., 1.-ny*ny - nz*nz))
    axis_x_y = ny
    axis_x_z = nz
    # Y脉冲轴
    axis_y_x = mx
    axis_y_y = np.sqrt(max(0., 1.-mx*mx - mz*mz))
    axis_y_z = mz

    sx_op = axis_x_x*sx + axis_x_y*sy + axis_x_z*sz
    sy_op = axis_y_x*sx + axis_y_y*sy + axis_y_z*sz

    def get_pulse_op(ptype):
        if ptype=='X':
            return sx_op, epsilon_x, t_pi_x
        elif ptype=='Y':
            return sy_op, epsilon_y, t_pi_y
        else:
            raise ValueError("Pulse must be X or Y")

    current_state = psi0
    all_times = [0.]
    all_states= [psi0]
    t_accum=0.

    for i in range(N):
        dt_free = intervals[i]
        if dt_free>1e-12:
            t_span = np.linspace(t_accum, t_accum+dt_free, N_free_step)
            H_free = [H0, H_noise]
            res_free = mesolve(H_free, current_state, t_span, [], [], 
                               options=Options(store_final_state=True))
            current_state = res_free.final_state
            t_accum += dt_free

            all_times.extend(t_span[1:])
            all_states.extend(res_free.states[1:])

        op_pulse, eps_pulse, t_pulse = get_pulse_op(pulses[i])
        amp = (np.pi + eps_pulse)/(2.* t_pulse) #2因子
        H_pulse = [H0, H_noise, [op_pulse, lambda t,args: amp]]

        t_span = np.linspace(t_accum, t_accum+t_pulse, N_pulse_step)
        res_pulse = mesolve(H_pulse, current_state, t_span, [], [],
                            options=Options(store_final_state=True))
        current_state = res_pulse.final_state
        t_accum += t_pulse
        all_times.extend(t_span[1:])
        all_states.extend(res_pulse.states[1:])

    dt_free = intervals[N]
    if dt_free>1e-12:
        t_span = np.linspace(t_accum, t_accum+dt_free, N_free_step)
        H_free = [H0, H_noise]
        res_free = mesolve(H_free, current_state, t_span, [], [],
                           options=Options(store_final_state=True))
        current_state = res_free.final_state
        t_accum += dt_free

        all_times.extend(t_span[1:])
        all_states.extend(res_free.states[1:])

    return np.array(all_times), all_states


##############################################################################
# (C) run_monte_carlo_general => 生成多次OU噪声, 做DD演化平均
##############################################################################
def run_monte_carlo_general(
    N_runs,
    T,
    pulses,
    t_pi_x, t_pi_y,
    epsilon_x, epsilon_y,
    ny, nz, mx, mz,
    tau_c, b,
    psi0,
    omega0,
    N_free_step,
    N_pulse_step
):
    N_x = sum([p=='X' for p in pulses])
    N_y = sum([p=='Y' for p in pulses])
    total_pulse_time = N_x*t_pi_x + N_y*t_pi_y
    if T< total_pulse_time:
        raise ValueError(f"T={T} < total_pulse_time={total_pulse_time}")

    N_noise_grid=2000
    ou_tlist= np.linspace(0, T, N_noise_grid)

    # 先跑一次: seed=0
    ou_noise_dummy= generate_ou_noise(ou_tlist, tau_c, b, B0=None, seed=0)
    test_times, test_states= run_single_experiment_general(
        T,pulses,
        t_pi_x,t_pi_y,
        epsilon_x,epsilon_y,
        ny,nz,mx,mz,
        psi0, omega0,
        ou_tlist, ou_noise_dummy,
        N_free_step, N_pulse_step
    )
    common_tlist= test_times
    N_points= len(common_tlist)

    sx, sy, sz= sigmax(), sigmay(), sigmaz()
    bloch_x_sum= np.zeros(N_points)
    bloch_y_sum= np.zeros(N_points)
    bloch_z_sum= np.zeros(N_points)

    for _ in range(N_runs):
        ou_noise= generate_ou_noise(ou_tlist, tau_c, b, B0=None)
        all_times, all_states= run_single_experiment_general(
            T, pulses,
            t_pi_x,t_pi_y,
            epsilon_x,epsilon_y,
            ny,nz,mx,mz,
            psi0,omega0,
            ou_tlist, ou_noise,
            N_free_step, N_pulse_step
        )
        for i in range(N_points):
            st= all_states[i]
            bloch_x_sum[i]+= expect(sx, st)
            bloch_y_sum[i]+= expect(sy, st)
            bloch_z_sum[i]+= expect(sz, st)

    bx_mean= bloch_x_sum/ N_runs
    by_mean= bloch_y_sum/ N_runs
    bz_mean= bloch_z_sum/ N_runs
    return common_tlist, bx_mean, by_mean, bz_mean

##############################################################################
# (D) single_T_job => 并行 map
##############################################################################
def single_T_job(
    T_val,
    N_runs,
    pulses,
    t_pi_x, t_pi_y,
    epsilon_x, epsilon_y,
    ny, nz, mx, mz,
    tau_c, b,
    psi0,
    omega0,
    N_free_step,
    N_pulse_step
):
    times,bx_mean,by_mean,bz_mean= run_monte_carlo_general(
        N_runs, T_val, pulses,
        t_pi_x, t_pi_y,epsilon_x, epsilon_y,
        ny,nz, mx,mz,
        tau_c, b,
        psi0,omega0,
        N_free_step,N_pulse_step
    )
    return bx_mean[-1]

##############################################################################
# (E) main => 并行扫描 T, 画图(包括 ln(-ln(C)) vs ln(T))
##############################################################################
if __name__=="__main__":
    #==== 1) 参数设定 ====
    tau_c= 100
    b= 30

    t_pi_x= 0.00001
    t_pi_y= 0.00001
    epsilon_x=0.0
    epsilon_y=0.0
    ny,nz= 0.0,0.0 #可以设定偏共振omega0，因此可以不用管nz
    mx,mz= 0.0,0.0

    m=4
    base_xy8= ['X','Y','X','Y','Y','X','Y','X']
    pulses= base_xy8 * m

    psi0= (basis(2,0)+basis(2,1)).unit()
    omega0= 0.0

    N_runs= 100
    N_free_step= 100
    N_pulse_step= 20

    #=== 2) 理论 T2 ===
    N_pulse= len(pulses)   # =8*m
    T2_theory= calc_theoretical_T2(N_pulse, tau_c, b)
    print(f"Using tau_c={tau_c}, b={b}, #pulses={N_pulse}")
    print(f"Theory T2= {T2_theory:.3f}")

    #=== 3) 扫描 T 值 ===
    N_x= sum(p=='X' for p in pulses)
    N_y= sum(p=='Y' for p in pulses)
    total_pulse_time= N_x*t_pi_x + N_y*t_pi_y
    T_min= 1000 * total_pulse_time
    T_max= 5
    NT= 11
    T_values= np.linspace(T_min, T_max, NT)

    # 并行
    job_func= partial(
        single_T_job,
        N_runs=N_runs,
        pulses=pulses,
        t_pi_x=t_pi_x, t_pi_y=t_pi_y,
        epsilon_x=epsilon_x, epsilon_y=epsilon_y,
        ny=ny, nz=nz, mx=mx, mz=mz,
        tau_c=tau_c, b=b,
        psi0=psi0, omega0=omega0,
        N_free_step=N_free_step,
        N_pulse_step=N_pulse_step
    )
    final_coherence_list= parallel_map(job_func, T_values)
    final_coherence_list= np.array(final_coherence_list)

    #=== 4) 拟合 Coherence => e^{-(T/T2)^3} ===
    def model_C(T, T2):
        return np.exp(- (T/T2)**3)

    popt, pcov= curve_fit(model_C, T_values, final_coherence_list, p0=[1.0])
    T2_fit= popt[0]
    T2_fit_err= np.sqrt(np.diag(pcov))[0]
    print(f"Fitted T2= {T2_fit:.3f} ± {T2_fit_err:.3f}")

    #=== 5) 画图1 => C vs T + 拟合
    T_dense= np.linspace(T_min, T_max, 200)
    C_fit= model_C(T_dense, T2_fit)

    plt.figure(figsize=(7,5))
    plt.plot(T_values, final_coherence_list, 'o', label="Data")
    label_str= (f"T2_fit={T2_fit:.3f}±{T2_fit_err:.3f}\nTheory={T2_theory:.3f}")
    plt.plot(T_dense, C_fit, 'r--', label=label_str)
    plt.xlabel("Total Time T")
    plt.ylabel("Final Coherence <X>")
    plt.title("DD + O-U noise, b-> sigma= sqrt(2)*b/sqrt(tau_c)")
    plt.grid(True)
    plt.legend()
    plt.show()

    #=== 6) 画图2 => ln(-ln(C)) vs ln(T)
    T_vals_plot=[]
    Y_vals_plot=[]
    for (Tv, Cf) in zip(T_values, final_coherence_list):
        if Cf>0 and Cf<1:
            val= - np.log(Cf)
            if val>0:
                T_vals_plot.append(Tv)
                Y_vals_plot.append(np.log(val))

    T_vals_plot= np.array(T_vals_plot)
    Y_vals_plot= np.array(Y_vals_plot)

    plt.figure(figsize=(6,5))
    xlog= np.log(T_vals_plot)
    ylog= Y_vals_plot
    plt.plot(xlog, ylog, 'o', label="Data")

    # 线性拟合 => y= a*x + b, 预期 a~3
    def linear_func(x,a,b):
        return a*x + b
    p_lin, pcov_lin= curve_fit(linear_func, xlog, ylog, p0=[3,0])
    a_lin, b_lin= p_lin
    da_lin, db_lin= np.sqrt(np.diag(pcov_lin))

    xfit= np.linspace(min(xlog), max(xlog),100)
    yfit= a_lin*xfit + b_lin
    plt.plot(xfit, yfit, 'r--', label=f"Slope= {a_lin:.2f}±{da_lin:.2f}")

    plt.xlabel("ln(T)")
    plt.ylabel("ln( -ln(C) )")
    plt.title("Check slope ~ 3 in log-log scale")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("In ln(-ln(C)) vs ln(T):")
    print(f"  slope= {a_lin:.3f}±{da_lin:.3f}, intercept= {b_lin:.3f}±{db_lin:.3f}")
    print("(If slope ~3 => data ~ e^{-(T/T2)^3}).")
