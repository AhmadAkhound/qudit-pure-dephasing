"""
Identifier: Code S3
Purpose: Verifies, via arbitrary-precision arithmetic (mpmath, 50 digits),
    that C(r) = sum_{k=1}^inf k*r^(k^2) is finite, positive, and monotonically
    increasing on r in (0,1), and confirms its asymptotic divergence
    C(r) ~ 1/[2(1-r)] as r -> 1^-.
Used in: Lemma on finiteness/positivity of C(r), following Section
    "Saturation scaling law".
Status: Valid and used in the manuscript.
"""
import numpy as np
from mpmath import mp, mpf

mp.dps = 50

def C_exact(r, kmax=20000):
    r = mpf(r)
    total = mpf(0)
    for k in range(1, kmax):
        term = k * r**(k**2)
        total += term
        if term < mpf(10)**(-60):   # قطع خودکار وقتی جمله‌ها ناچیز شدند
            break
    return total

if __name__ == "__main__":
    # (الف) و (ب): همگرایی و یکنوایی -- بررسی روی طیف وسیعی از r
    print("=== (الف),(ب) همگرایی و یکنوایی C(r) ===")
    rs_monotonic = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999]
    prev = None
    monotonic_ok = True
    for r in rs_monotonic:
        val = C_exact(r)
        increasing = (prev is None) or (val > prev)
        monotonic_ok = monotonic_ok and increasing
        print(f"  r={r:6.3f}  C(r)={float(val):.6f}  یکنوا_تا_اینجا={monotonic_ok}")
        prev = val

    # (پ) رفتار مجانبی نزدیک r=1
    print("\n=== (پ) رفتار مجانبی C(r) ~ 1/(2(1-r)) نزدیک r->1 ===")
    rs_asymp = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
    print(f"{'r':>10} {'C(r) دقیق':>16} {'1/(2(1-r))':>16} {'نسبت C(r)*(1-r)':>18}")
    for r in rs_asymp:
        val = C_exact(r)
        asym = mpf(1) / (2*(1-mpf(r)))
        ratio = val * (1 - mpf(r))
        print(f"{r:10.6f} {float(val):16.4f} {float(asym):16.4f} {float(ratio):18.6f}  (باید -> 0.5)")