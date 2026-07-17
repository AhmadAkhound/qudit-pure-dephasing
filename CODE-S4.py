"""
Identifier: Code S4
Purpose: Verifies the rigorous error bound |R(d)| <= r^(d^2)[1/(1-r^(2d)) +
    r^(2d)/(d(1-r^(2d))^2)] of the convergence theorem, using arbitrary-
    precision arithmetic (mpmath, 2500 digits) to avoid catastrophic
    cancellation at double precision (the bound is as small as ~1e-1922
    for d=80, r~0.5).
Used in: Section "Saturation scaling law", numerical verification of the
    convergence theorem's error bound.
Status: Valid and used in the manuscript. Tested on 36 independent (r,d)
    combinations drawn from the six physical parameter regimes of the paper;
    zero violations found.
"""
from mpmath import mp, mpf

mp.dps = 2500  # دقت بسیار بالا، برای حل کوچک‌ترین کران‌های آزموده‌شده (~1e-1900)

def negativity_formula_hp(r, d):
    r = mpf(r)
    return (mpf(1)/d) * sum((d-k) * r**(k**2) for k in range(1, d))

def negativity_inf_hp(r, kmax=200):
    r = mpf(r)
    return sum(r**(k**2) for k in range(1, kmax))

def C_of_r_hp(r, kmax=200):
    r = mpf(r)
    return sum(k * r**(k**2) for k in range(1, kmax))

def R_bound_hp(r, d):
    r = mpf(r)
    return r**(d**2) * (1/(1-r**(2*d)) + r**(2*d)/(d*(1-r**(2*d))**2))

if __name__ == "__main__":
    # همان مقادیر r که در پروژه از رژیم‌های پارامتری واقعی (CODE-06) به‌دست آمده‌اند
    test_rs = [0.834, 0.920, 0.977, 0.613, 0.501, 0.676]
    dims = [5, 10, 20, 30, 50, 80]

    all_ok = True
    violations = []
    total = 0
    for r in test_rs:
        Ninf = negativity_inf_hp(r)
        C = C_of_r_hp(r)
        for d in dims:
            Nd = negativity_formula_hp(r, d)
            actual_R = abs((Ninf - Nd) - C/d)
            bound = R_bound_hp(r, d)
            ok = bound >= actual_R
            total += 1
            if not ok:
                violations.append((r, d, actual_R, bound))
            all_ok = all_ok and ok

    print(f"تعداد کل چک‌ها: {total}   تعداد نقض واقعی: {len(violations)}")
    for v in violations:
        print("  نقض:", v)
    print(f"\n>>> نتیجه‌ی نهایی: {'کران در تمام موارد برقرار است (دقت 2500 رقمی)' if all_ok else 'نقض یافت شد - نیاز به بررسی بیشتر'}")