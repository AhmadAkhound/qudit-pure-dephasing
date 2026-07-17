"""
Identifier: Code S10
Purpose: (a) Verifies the closed-form decoherence function
    Gamma_Ohmic(t) = (eta/2) ln(1+omega_c^2 t^2) [Leggett et al., Rev. Mod.
    Phys. 59, 1 (1987)] against direct numerical integration of
    Gamma(t) = int J(w)/w^2 (1-cos wt) dw with J(w) = eta*w*exp(-w/w_c).
    (b) Tests the 1/d saturation identity at four arbitrary time points
    (rather than peak/trough, since Gamma_Ohmic(t) is monotonic and
    non-oscillatory), for d=20 to d=400.
Used in: Section "Independence from spectral-density class: the Ohmic
    case" (both tables).
Status: Valid and used in the manuscript. Full agreement to 8 significant
    digits (part a); fit residuals at machine precision, ~1e-16 (part b).
"""
import numpy as np
from scipy import integrate, optimize

def Gamma_Ohmic_numeric(t, eta=0.15, omega_c=1.0):
    def integrand(w):
        J = eta * w * np.exp(-w/omega_c)
        return J / w**2 * (1 - np.cos(w * t))
    val, _ = integrate.quad(integrand, 1e-8, 200*omega_c, limit=800)
    return val

def Gamma_Ohmic_closed(t, eta=0.15, omega_c=1.0):
    return (eta/2) * np.log(1 + (omega_c*t)**2)

def r_ohmic(t, **kw):
    return np.exp(-2*Gamma_Ohmic_closed(t, **kw))

def negativity_formula(r, d):
    k = np.arange(1, d)
    return (1/d) * np.sum((d-k) * r**(k**2))

def negativity_inf(r, kmax=3000):
    k = np.arange(1, kmax)
    return np.sum(r**(k**2))

if __name__ == "__main__":
    print("=== (الف) تطبیق Gamma_Ohmic: عددی در برابر فرمول بسته ===")
    for t in [0.5, 1.0, 2.0, 5.0, 10.0]:
        g_num = Gamma_Ohmic_numeric(t)
        g_closed = Gamma_Ohmic_closed(t)
        match = abs(g_num - g_closed) < 1e-5
        print(f"  t={t:5.1f}  Gamma_numeric={g_num:.8f}  Gamma_closed={g_closed:.8f}  تطابق={match}")

    print("\n=== (ب) آزمون قانون اشباع 1/d در نقاط زمانی دلخواه (Ohmic) ===")
    eta, omega_c = 0.15, 1.0
    dims = np.array([20, 30, 50, 80, 120, 180, 260, 400])
    for t_test in [1.0, 3.0, 5.0, 10.0]:
        r_t = r_ohmic(t_test, eta=eta, omega_c=omega_c)
        N_inf = negativity_inf(r_t)
        amps = np.array([negativity_formula(r_t, d) for d in dims])

        def model(d, C):
            return N_inf - C/d
        popt, _ = optimize.curve_fit(model, dims, amps, p0=[1.0])
        fit_resid = amps - model(dims, *popt)
        print(f"\n  t={t_test}: r(t)={r_t:.6f}  N_inf={N_inf:.6f}")
        print(f"    C(برازش)={popt[0]:.6f}   max|fit_residual|={np.max(np.abs(fit_resid)):.2e}")