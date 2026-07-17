"""
Identifier: Code S8
Purpose: Rigorous test of the 1/d saturation law for negativity, verifying
    that [N_infinity - N(r,d)] * d converges to a constant across six
    independent environmental parameter regimes (weak/base/strong coupling;
    narrow/base/broad non-Markovian bandwidth; two central frequencies).
    Peak/trough values of r(t) are first located via a coarse global scan
    (4000 points) before local refinement, to guarantee the true first
    extremum is used. Dimensions tested: d=10 to d=200. Includes automatic
    PASS/FAIL assertion at a 1e-4% relative-deviation threshold.
Used in: Section "Saturation scaling law", rigorous numerical verification
    subsection (Table of six regimes).
Status: Valid and used in the manuscript. PASS in all six regimes, with
    relative deviation of 0.00000% between d=100,150,200.
"""
import numpy as np
from scipy import integrate, optimize

def Gamma(t, omega0, lam, eta):
    def integrand(w):
        J = eta * lam**2 / ((w - omega0)**2 + lam**2)
        return J / w**2 * (1 - np.cos(w * t))
    val, _ = integrate.quad(integrand, 1e-6, 50*omega0, limit=400)
    return val

def r_of_t(t, omega0, lam, eta):
    return np.exp(-2*Gamma(t, omega0, lam, eta))

def negativity_formula(r, d):
    k = np.arange(1, d)
    return (1/d) * np.sum((d-k) * r**(k**2))

def negativity_inf(r, kmax=3000):
    k = np.arange(1, kmax)
    return np.sum(r**(k**2))

def find_peak_trough_r(omega0, lam, eta):
    # اسکن درشت اولیه روی بازه‌ی وسیع، برای توجیه این‌که بازه‌های ریزبینانه‌ی
    # بعدی واقعاً حاوی اولین کمینه/بیشینه‌ی سراسری‌اند، نه فقط یک اکسترمم محلی حدسی
    t_scan = np.linspace(0.05, 20, 4000)
    r_scan = np.array([r_of_t(t, omega0, lam, eta) for t in t_scan])
    from scipy.signal import argrelextrema
    minima_idx = argrelextrema(r_scan, np.less)[0]
    maxima_idx = argrelextrema(r_scan, np.greater)[0]
    assert len(minima_idx) > 0 and len(maxima_idx) > 0, "اسکن درشت هیچ اکسترممی پیدا نکرد"
    t_trough_coarse = t_scan[minima_idx[0]]
    t_peak_coarse = t_scan[maxima_idx[0]]

    f = lambda t: -r_of_t(t, omega0, lam, eta)
    g = lambda t: r_of_t(t, omega0, lam, eta)
    win = 0.5
    res_trough = optimize.minimize_scalar(g, bounds=(t_trough_coarse-win, t_trough_coarse+win), method='bounded')
    res_peak   = optimize.minimize_scalar(f, bounds=(t_peak_coarse-win, t_peak_coarse+win), method='bounded')
    return -res_peak.fun, res_trough.fun

if __name__ == "__main__":
    param_sets = [
        dict(omega0=1.0, lam=0.1, eta=0.15),
        dict(omega0=1.0, lam=0.1, eta=0.05),
        dict(omega0=1.0, lam=0.1, eta=0.35),
        dict(omega0=1.0, lam=0.05, eta=0.15),
        dict(omega0=1.0, lam=0.25, eta=0.15),
        dict(omega0=2.0, lam=0.2, eta=0.15),
    ]
    dims = [10, 15, 20, 30, 50, 80, 100, 150, 200]
    for pset in param_sets:
        r_peak, r_trough = find_peak_trough_r(**pset)
        amp_inf = negativity_inf(r_peak) - negativity_inf(r_trough)
        print(f"\n--- {pset}  r_peak={r_peak:.5f} r_trough={r_trough:.5f} amp_inf={amp_inf:.5f} ---")
        Cvals = []
        for d in dims:
            amp_d = negativity_formula(r_peak, d) - negativity_formula(r_trough, d)
            C_est = (amp_inf - amp_d) * d
            Cvals.append(C_est)
            print(f"  d={d:3d}  amp(d)={amp_d:.5f}  resid*d={C_est:.5f}")
        Cvals = np.array(Cvals)
        rel = (Cvals[-3:].max()-Cvals[-3:].min())/Cvals[-3:].mean()*100
        status = "PASS" if rel < 1e-4 else "FAIL"
        print(f"  --> پایداری C: میانگین={Cvals[-3:].mean():.5f}  انحراف نسبی={rel:.5f}%   [{status}, آستانه=1e-4%]")
        assert rel < 1e-4, f"نقض ادعای مقاله برای {pset}: انحراف نسبی {rel}% >= 1e-4%"