"""
=============================================================================
**s = 1/2 拉比振荡实验 + O-U(Ornstein-Uhlenbeck) 噪声 并行模拟**

主要特点:
1) generate_ou_noise: 以 b 做稳态方差 sqrt, 由 sigma = sqrt(2)*b / sqrt(tau_c),
   使用 Exact Discretization 公式生成 O-U 过程.
2) 最终画图包含:
   - Transition Probability vs T + 拟合 cos(2*pi*f*T) * e^(-T/td)，td为噪声造成的拉比振幅衰减特征时间。

=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip.parallel import parallel_map
from scipy.optimize import curve_fit
from functools import partial

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
# (B) 单次实验: 拉比振荡实验
##############################################################################
def run_single_experiment_larmor(
    T,
    t_pi,
    psi0,
    omega0,
    ou_tlist,
    ou_noise,
    N_pulse_step
):
    """
    单次拉比振荡实验：
    - 使用一个简单的X轴脉冲（因为只有幅度变化），
    - O-U噪声和恒定的偏共振 omega0。
    """
    sx, sy, sz = sigmax(), sigmay(), sigmaz()
    H0 = (omega0 / 2) * sz

    def beta_t(t, args=None):
        if t < 0:
            return 0.
        if t >= ou_tlist[-1]:
            return ou_noise[-1]
        idx = np.searchsorted(ou_tlist, t)
        if idx == 0:
            return ou_noise[0]
        if idx >= len(ou_tlist):
            return ou_noise[-1]
        t1 = ou_tlist[idx-1]
        t2 = ou_tlist[idx]
        f1 = ou_noise[idx-1]
        f2 = ou_noise[idx]
        return f1 + (f2 - f1) * ((t - t1) / (t2 - t1))

    H_noise = [sz / 2.0, beta_t]

    # X脉冲轴
    axis_x_x = 1.0
    sx_op = axis_x_x * sx

    # 给定的幅度和持续时间
    amp = np.pi / (2. * t_pi)
    H_pulse = [H0, H_noise, [sx_op, lambda t, args: amp]]

    current_state = psi0
    all_times = [0.]
    all_states = [psi0]

    t_span = np.linspace(0, T, N_pulse_step)
    res_pulse = mesolve(H_pulse, current_state, t_span, [], [],
                        options=Options(store_final_state=True))
    current_state = res_pulse.final_state
    all_times.extend(t_span[1:])
    all_states.extend(res_pulse.states[1:])

    return np.array(all_times), all_states

##############################################################################
# (C) run_monte_carlo_general => 生成多次OU噪声, 做DD演化平均
##############################################################################
def run_monte_carlo_general(
    N_runs,
    T,
    t_pi,
    psi0,
    omega0,
    tau_c, b,
    N_pulse_step
):
    N_noise_grid = 2000
    ou_tlist = np.linspace(0, T, N_noise_grid)

    # 先跑一次: seed=0
    ou_noise_dummy = generate_ou_noise(ou_tlist, tau_c, b, B0=None, seed=0)
    test_times, test_states = run_single_experiment_larmor(
        T, t_pi, psi0, omega0,
        ou_tlist, ou_noise_dummy, N_pulse_step
    )
    common_tlist = test_times
    N_points = len(common_tlist)

    # Calculate Transition Probability
    prob_sum = np.zeros(N_points)

    for _ in range(N_runs):
        ou_noise = generate_ou_noise(ou_tlist, tau_c, b, B0=None)
        all_times, all_states = run_single_experiment_larmor(
            T, t_pi, psi0, omega0,
            ou_tlist, ou_noise, N_pulse_step
        )
        for i in range(N_points):
            st = all_states[i]
            prob_sum[i] += abs(expect(basis(2, 0) * basis(2, 0).dag(), st))**2  # 末态在0态的概率

    prob_mean = prob_sum / N_runs
    return common_tlist, prob_mean

##############################################################################
# (D) single_T_job => 并行 map
##############################################################################
def single_T_job(
    T_val,
    N_runs,
    t_pi,
    psi0,
    omega0,
    tau_c, b,
    N_pulse_step
):
    times, prob_mean = run_monte_carlo_general(
        N_runs, T_val, t_pi, psi0, omega0, tau_c, b, N_pulse_step
    )
    return prob_mean[-1]

##############################################################################
# (E) main => 并行扫描 T, 画图(包括 ln(-ln(C)) vs ln(T))
##############################################################################
if __name__ == "__main__":
    #==== 1) 参数设定 ====
    tau_c = 10.0
    b = 3.0

    t_pi = 1  # 设定的 π 脉冲持续时间，用于换算amp
    psi0 = basis(2, 0)  # 初始态 |0>
    omega0 = 0.0  # 偏共振

    N_runs = 100
    N_pulse_step = 2000

    #=== 2) 扫描 T 值 ===
    T_min = 0.1
    T_max = 3
    NT = 30
    T_values = np.linspace(T_min, T_max, NT)

    # 并行计算
    job_func = partial(
        single_T_job,
        N_runs=N_runs,
        t_pi=t_pi,
        psi0=psi0, omega0=omega0,
        tau_c=tau_c, b=b,
        N_pulse_step=N_pulse_step
    )
    final_prob_list = parallel_map(job_func, T_values)
    final_prob_list = np.array(final_prob_list)

    #=== 3) 拟合 Probability => cos(2*pi*f*T) * e^(-T/td) ===
    def model_prob(T, f, td):
        return 0.5 * (1 + np.cos(2*np.pi*f*T) * np.exp(-T/td))

    popt, pcov = curve_fit(model_prob, T_values, final_prob_list, p0=[1.0, 20.0])
    f_fit, td_fit = popt
    td_fit_err = np.sqrt(np.diag(pcov))[1]
    print(f"Fitted td = {td_fit:.3f} ± {td_fit_err:.3f}")

    #=== 4) 画图1 => Probability vs T + 拟合
    T_dense = np.linspace(T_min, T_max, 200)
    prob_fit = model_prob(T_dense, f_fit, td_fit)

    plt.figure(figsize=(7, 5))
    plt.plot(T_values, final_prob_list, 'o', label="Data")
    plt.plot(T_dense, prob_fit, 'r--', label=f"Fitted td={td_fit:.3f}±{td_fit_err:.3f}")
    plt.xlabel("Total Time T")
    plt.ylabel("Transition Probability")
    plt.title("Larmor Oscillation + O-U Noise")
    plt.grid(True)
    plt.legend()
    plt.show()
