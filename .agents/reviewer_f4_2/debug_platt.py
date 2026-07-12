import numpy as np

def fit_platt_scaling_correct(dec_values, y):
    N = len(y)
    N_plus = np.sum(y > 0)
    N_minus = np.sum(y <= 0)
    
    t_plus = (N_plus + 1.0) / (N_plus + 2.0)
    t_minus = 1.0 / (N_minus + 2.0)
    t = np.where(y > 0, t_plus, t_minus)
    
    A = 0.0
    B = np.log((N_minus + 1.0) / (N_plus + 1.0))
    
    for sign in ['+', '-']:
        curr_A, curr_B = A, B
        print(f"\nTesting with sign: {sign}")
        for iter_idx in range(10):
            gA, gB = 0.0, 0.0
            HAA, HAB, HBB = 0.0, 0.0, 0.0
            
            nll = 0.0
            for i in range(N):
                f = dec_values[i]
                arg = curr_A * f + curr_B
                exp_val = np.exp(np.clip(arg, -50, 50))
                p = 1.0 / (1.0 + exp_val)
                
                ln_p = -np.log(1.0 + exp_val)
                ln_1_p = arg - np.log(1.0 + exp_val)
                nll -= (t[i] * ln_p + (1.0 - t[i]) * ln_1_p)
                
                d = t[i] - p
                gA += d * f
                gB += d
                
                p_1_p = p * (1.0 - p)
                HAA += p_1_p * f * f
                HAB += p_1_p * f
                HBB += p_1_p
                
            HAA += 1e-5
            HBB += 1e-5
            det = HAA * HBB - HAB * HAB
            
            dA = (HBB * gA - HAB * gB) / det
            dB = (-HAB * gA + HAA * gB) / det
            
            print(f"  Iter {iter_idx}: NLL = {nll:.6f}, gA = {gA:.6f}, gB = {gB:.6f}, dA = {dA:.6f}, dB = {dB:.6f}, A = {curr_A:.6f}, B = {curr_B:.6f}")
            
            if sign == '+':
                curr_A += dA
                curr_B += dB
            else:
                curr_A -= dA
                curr_B -= dB

np.random.seed(42)
dec = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
y = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
fit_platt_scaling_correct(dec, y)
