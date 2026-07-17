"""
Identifier: Code S9
Purpose: Extends the 1/d saturation test to discord, by directly fitting the
    two-parameter model Delta D(d) = A - C/d (and the extended model with a
    C2/d^2 term) via nonlinear regression on d=20 to d=4000, computed
    directly from the closed-form discord formula (no numerical
    optimization), across the same six environmental parameter regimes as
    Code S6. Gamma(t) is integrated with tightened quadrature tolerances
    (epsabs=epsrel=1e-14).
Used in: Section "Generalizing the saturation law to discord" (Table of six
    regimes, fit parameters A, C with uncertainties, RMS/max residuals).
Status: Valid and used in the manuscript. In four of six regimes, C2 is
    statistically consistent with zero; in two regimes with the largest
    r_peak (weak coupling, narrow bandwidth), C2 is small but statistically
    nonzero (~1e-3, several standard errors from zero), indicating a
    possible weak O(d^-2) correction not analytically established for
    discord (unlike negativity, for which Theorem 1 proves double-
    exponential convergence).
"""
import numpy as np
from scipy import integrate, optimize

def Gamma(t, omega0, lam, eta):
    def integrand(w):
        J = eta * lam**2 / ((w - omega0)**2 + lam**2)
        return J / w**2 * (1 - np.cos(w * t))
    val, _ = integrate.quad(integrand, 1e-6, 50*omega0, limit=400, epsabs=1e-14, epsrel=1e-14)
    return val

def r_of_t(t, omega0, lam, eta):
    return np.exp(-2*Gamma(t, omega0, lam, eta))

def entropy_toeplitz(r, d):
    k = np.arange(d)
    M = r**((k[:,None]-k[None,:])**2) / d
    eigs = np.linalg.eigvalsh(M)
    eigs = eigs[eigs>1e-14]
    return -np.sum(eigs*np.log(eigs))

def discord_closed_form(r, d):
    return np.log(d) - entropy_toeplitz(r, d)

def find_peak_trough_r(omega0, lam, eta):
    # اسکن درشت اولیه (مشابه CODE-06 نسخه‌ی ۲) تا اعتبار بازه‌های ریزبینانه تضمین شود
    t_scan = np.linspace(0.05, 20, 4000)
    r_scan = np.array([r_of_t(t, omega0, lam, eta) for t in t_scan])
    from scipy.signal import argrelextrema
    minima_idx = argrelextrema(r_scan, np.less)[0]
    maxima_idx = argrelextrema(r_scan, np.greater)[0]
    t_trough_coarse = t_scan[minima_idx[0]]
    t_peak_coarse = t_scan[maxima_idx[0]]

    f = lambda t: -r_of_t(t, omega0, lam, eta)
    g = lambda t: r_of_t(t, omega0, lam, eta)
    win = 0.5
    res_trough = optimize.minimize_scalar(g, bounds=(t_trough_coarse-win, t_trough_coarse+win), method='bounded')
    res_peak   = optimize.minimize_scalar(f, bounds=(t_peak_coarse-win, t_peak_coarse+win), method='bounded')
    return -res_peak.fun, res_trough.fun
def model(d, A, C):
    return A - C/d

def model2(d, A, C, C2):
    return A - C/d - C2/d**2

if __name__ == "__main__":
    param_sets = [
        dict(omega0=1.0, lam=0.1, eta=0.15),
        dict(omega0=1.0, lam=0.1, eta=0.05),
        dict(omega0=1.0, lam=0.1, eta=0.35),
        dict(omega0=1.0, lam=0.05, eta=0.15),
        dict(omega0=1.0, lam=0.25, eta=0.15),
        dict(omega0=2.0, lam=0.2, eta=0.15),
    ]
    dims = np.array([20, 30, 50, 80, 120, 180, 260, 400, 600, 900, 1500, 2500, 4000])

    for pset in param_sets:
        r_peak, r_trough = find_peak_trough_r(**pset)
        amps = np.array([discord_closed_form(r_peak, d) - discord_closed_form(r_trough, d)
                          for d in dims])
        popt, pcov = optimize.curve_fit(model, dims, amps, p0=[amps[-1], 1.0])
        A_fit, C_fit = popt
        A_err, C_err = np.sqrt(np.diag(pcov))
        resid = amps - model(dims, *popt)
        rms_resid = np.sqrt(np.mean(resid**2))
        popt2, pcov2 = optimize.curve_fit(model2, dims, amps, p0=[amps[-1], 1.0, 0.0])
        C2_fit = popt2[2]
        C2_err = np.sqrt(np.diag(pcov2))[2]
        print(f"\n--- {pset}  r_peak={r_peak:.5f} r_trough={r_trough:.5f} ---")
        print(f"  A={A_fit:.6f}±{A_err:.2e}  C={C_fit:.6f}±{C_err:.2e}")
        print(f"  max|residual|={np.max(np.abs(resid)):.2e}  RMS residual={rms_resid:.2e}")
        print(f"  مدل گسترش‌یافته A-C/d-C2/d^2: C2={C2_fit:.4e}±{C2_err:.2e} (باید ~0 باشد)")
