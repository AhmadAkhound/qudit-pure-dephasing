"""
Identifier: Code S1 
Purpose: Cross-checks the closed-form negativity formula (negativity_formula)
    against direct numerical partial transpose (negativity_direct), for
    d=2,3,4 with a Lorentzian spectral density (eta=1.0, relatively strong
    coupling), at three independent time points.
Used in: Section "Closed-form negativity formula", Eq. (formula vs. direct
    check at t=0.5, 2.0, 5.0).
Status: Valid and used in the manuscript. Model parameters: omega0=1.0,
    lam=0.1, eta=1.0 (Lorentzian spectral density, T=0).
"""
import numpy as np
from scipy import integrate
from scipy.signal import argrelextrema

def Gamma(t, omega0=1.0, lam=0.1, eta=1.0):
    def integrand(w):
        J = eta * lam**2 / ((w - omega0)**2 + lam**2)
        return J / w**2 * (1 - np.cos(w * t))
    val, _ = integrate.quad(integrand, 1e-6, 50*omega0, limit=400)
    return val

def r_of_t(t, **kw):
    return np.exp(-2*Gamma(t, **kw))

def negativity_formula(r, d):
    k = np.arange(1, d)
    return (1/d) * np.sum((d-k) * r**(k**2))

def negativity_direct(r, d, phi=None):
    if phi is None:
        phi = np.zeros((d,d))
    dim = d*d
    rho = np.zeros((dim,dim), dtype=complex)
    idx = lambda n: n*d+n
    for n in range(d):
        for m in range(d):
            rho[idx(n), idx(m)] = (1/d)*r**((n-m)**2)*np.exp(1j*phi[n,m])
    rho_pt = rho.reshape(d,d,d,d).transpose(0,3,2,1).reshape(dim,dim)
    eigs = np.linalg.eigvalsh(rho_pt)
    return np.sum(np.abs(eigs[eigs<0]))

if __name__ == "__main__":
    ts = np.linspace(0.01, 15, 300)
    dims = [2,3,4]
    for d in dims:
        print(f"\n=== d={d} ===")
        for t_test in [0.5, 2.0, 5.0]:
            r_test = r_of_t(t_test)
            print(f"  check t={t_test}: formula={negativity_formula(r_test,d):.6f}  "
                  f"direct={negativity_direct(r_test,d):.6f}")
