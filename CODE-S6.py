"""
Identifier: Code S6
Purpose: Compares three approaches to the near-r=1 behavior of C(r) =
    sum_{k=1}^inf k*r^(k^2): (a) exact high-precision series summation
    (mpmath), (b) a continuous Gaussian-integral approximation using the
    exact prefactor a=-ln(r) [rather than the linearized a=1-r], and (c) the
    leading-order asymptotic estimate C(r)~1/[2(1-r)] stated in the main
    text. Shows that using -ln(r) instead of (1-r) improves the relative
    error by roughly a factor of 4 across the tested range r in [0.1,0.9999].
Used in: Remark following Lemma 1 (finiteness/positivity of C(r)),
    clarifying that the asymptotic estimate is for interpretation only and
    is not used in the proof of Theorem 1.
Status: Valid, supplementary precision check.
"""
import numpy as np
from mpmath import mp

mp.dps = 80

# ---------- exact C(r) ----------
def C_exact(r):
    r = mp.mpf(r)
    s = mp.mpf('0')
    k = 1
    while True:
        term = k*r**(k*k)
        s += term
        if term < mp.mpf('1e-60'):
            break
        k += 1
    return s

# ---------- asymptotic ----------
def C_asym(r):
    return 1/(2*(1-r))

# ---------- integral (exact prefactor a = -ln r) ----------
def C_integral(r):
    a = -mp.log(r)
    f = lambda x: x*mp.e**(-a*x*x)
    return mp.quad(f, [0, mp.inf])

# ---------- numerical derivative ----------
def dC_numeric(r):
    h = mp.mpf("1e-6")
    return (C_exact(r+h)-C_exact(r-h))/(2*h)

if __name__ == "__main__":
    rs = [0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90,
          0.93,0.95,0.97,0.98,0.99,0.995,0.999,0.9995,0.9999]

    print("="*130)
    print(f"{'r':>8} {'C(r)':>18} {'Integral':>18} {'Asymptotic':>18} {'C*(1-r)':>14} {'Rel.Err(Int)':>14} {'Rel.Err(Asym)':>14} {'dC/dr':>18}")
    print("="*130)

    for r in rs:
        C = C_exact(r)
        I = C_integral(r)
        A = C_asym(r)
        errI = abs((I-C)/C)
        errA = abs((A-C)/C)
        der = dC_numeric(mp.mpf(r))

        print(f"{r:8.4f}"
              f"{float(C):18.8f}"
              f"{float(I):18.8f}"
              f"{float(A):18.8f}"
              f"{float(C*(1-r)):14.8f}"
              f"{float(errI):14.3e}"
              f"{float(errA):14.3e}"
              f"{float(der):18.8f}")

    print("="*130)