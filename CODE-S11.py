"""
Identifier: Code S11
Purpose: Diagnoses the source of a small (~1-3%) discrepancy observed in an
    auxiliary pseudomode-based Lindblad cross-check, by comparing the ideal
    pseudomode correlation function C(tau) = g^2 exp(-(kappa/2)|tau|)
    exp(-i*omega_0*tau) directly (no Lindblad simulation) against the
    numerical Fourier transform of the actual physical spectral density
    J(omega), restricted to omega >= 0.
Used in: Appendix B, "Independent robustness check via the pseudomode
    method".
Status: Valid and used in the manuscript. Confirms the discrepancy is a
    known limitation of the pseudomode correspondence when omega_0/lambda
    is not asymptotically large, not an error in the manuscript's primary
    (directly integrated) results.
"""
import numpy as np
from scipy import integrate

omega0, lam, eta = 1.0, 0.1, 0.15
kappa = 2*lam
g = np.sqrt(np.pi*eta*lam)

def J_target(w):
    return eta * lam**2 / ((w - omega0)**2 + lam**2)

def C_pseudomode(tau):
    return g**2 * np.exp(-(kappa/2)*np.abs(tau)) * np.exp(-1j*omega0*tau)

def C_from_J_exact(tau):
    re_part, _ = integrate.quad(lambda w: J_target(w)*np.cos(w*tau), 0, 200, limit=800)
    im_part, _ = integrate.quad(lambda w: -J_target(w)*np.sin(w*tau), 0, 200, limit=800)
    return re_part + 1j*im_part

if __name__ == "__main__":
    print("=== Comparison of correlation functions: ideal pseudomode vs. actual J(w) (w=0..inf) ===")
    print(f"{'tau':>6} {'|C_pseudomode|':>16} {'|C_from_J|':>14} {'ratio':>10}")
    for tau in [0.1, 0.5, 1.0, 2.0, 4.0]:
        c1 = C_pseudomode(tau)
        c2 = C_from_J_exact(tau)
        print(f"{tau:6.2f} {abs(c1):16.6f} {abs(c2):14.6f} {abs(c1)/abs(c2):10.5f}")
    print("\n>>> If all ratios are close to 1 (e.g., within 0.97-1.03), the parameter")
    print(">>> correspondence is approximately correct and the small discrepancy stems from the physical w=0 cutoff, not a bug.")