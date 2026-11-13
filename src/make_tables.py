"""
make_tables.py
Turns the results CSVs into LaTeX tables for the report, so that no number in
the write-up is copied by hand.

CS 4033/5033 Machine Learning Fundamentals, Fall 2026, Project 2
Author: Muhammad Ridwanul Hoque
"""

import os
import csv
import numpy as np

from experiment import RESULTS, DIMS, NOISE, NOISE_NAME, load

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report", "tables")


def rd(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return list(csv.DictReader(fh))


def fmt_eq(est, se):
    """y-hat = b0 (se) + b1 (se) x1 ..., as LaTeX."""
    names = ["x"] if len(est) == 2 else [f"x_{i}" for i in range(1, len(est))]
    s = f"{est[0]:.3f}_{{({se[0]:.3f})}}"
    for b, e, nm in zip(est[1:], se[1:], names):
        s += f" {'-' if b < 0 else '+'} {abs(b):.3f}_{{({e:.3f})}}\\,{nm}"
    return s


def main():
    os.makedirs(OUT, exist_ok=True)
    summ = {r["dataset"]: r for r in rd("summary.csv")}
    coefs = rd("coefficients.csv")

    # Table: fitted equations
    lines = []
    for d in DIMS:
        for nz in NOISE:
            name = f"{d}_{nz}"
            rows = [r for r in coefs if r["dataset"] == name]
            est = [float(r["estimate"]) for r in rows]
            se = [float(r["se"]) for r in rows]
            lines.append(f"{d} {NOISE_NAME[nz]} & $\\hat y = {fmt_eq(est, se)}$ \\\\")
        if d != "3D":
            lines.append("\\midrule")
    open(os.path.join(OUT, "equations.tex"), "w").write("\n".join(lines) + "\n")

    # Table: fit metrics
    lines = []
    for d in DIMS:
        for nz in NOISE:
            r = summ[f"{d}_{nz}"]
            f = lambda k, n=3: f"{float(r[k]):.{n}f}"
            lines.append(
                f"{d} {NOISE_NAME[nz]} & {f('R2')} & {f('adj_R2')} & {f('RMSE', 4)} & {f('MAE', 4)} & "
                f"{f('sigma_hat', 4)} & {float(r['F']):.1f} & {float(r['F_p']):.1e} \\\\")
        if d != "3D":
            lines.append("\\midrule")
    open(os.path.join(OUT, "metrics.tex"), "w").write("\n".join(lines) + "\n")

    # Table: generalization, with the theoretical expectation
    lines = []
    for d in DIMS:
        for nz in NOISE:
            r = summ[f"{d}_{nz}"]
            p, n = int(r["dims"]), int(r["n"])
            s = float(r["sigma_hat"])
            expected = s * np.sqrt(1 + (p + 1) / n)
            rm, cv = float(r["RMSE"]), float(r["CV_RMSE"])
            lines.append(
                f"{d} {NOISE_NAME[nz]} & {rm:.4f} & {cv:.4f} $\\pm$ {float(r['CV_RMSE_sd']):.4f} & "
                f"{100 * (cv / rm - 1):.1f}\\% & {expected:.4f} & {float(r['R2']):.3f} & "
                f"{float(r['CV_R2']):.3f} \\\\")
        if d != "3D":
            lines.append("\\midrule")
    open(os.path.join(OUT, "generalization.tex"), "w").write("\n".join(lines) + "\n")

    # Table: assumption checks
    lines = []
    for d in DIMS:
        for nz in NOISE:
            r = summ[f"{d}_{nz}"]
            lines.append(f"{d} {NOISE_NAME[nz]} & {float(r['shapiro_W']):.3f} & {float(r['shapiro_p']):.2f} & "
                         f"{float(r['BP_LM']):.2f} & {float(r['BP_p']):.2f} \\\\")
        if d != "3D":
            lines.append("\\midrule")
    open(os.path.join(OUT, "assumptions.tex"), "w").write("\n".join(lines) + "\n")

    # Table: full coefficient inference
    lines = []
    prev = None
    for r in coefs:
        d, nz = r["dataset"].split("_")
        label = f"{d} {NOISE_NAME[nz]}" if r["dataset"] != prev else ""
        if prev is not None and r["dataset"] != prev:
            lines.append("\\addlinespace[2pt]")
        prev = r["dataset"]
        term = r["term"].split()[0]
        term = {"b0": "$b_0$", "b1": "$b_1$", "b2": "$b_2$", "b3": "$b_3$"}[term]
        p = float(r["p"])
        ptxt = "$<10^{-6}$" if p < 1e-6 else f"{p:.4f}"
        lines.append(f"{label} & {term} & {float(r['estimate']):.4f} & {float(r['se']):.4f} & "
                     f"{float(r['t']):.2f} & {ptxt} & [{float(r['ci95_low']):.3f}, {float(r['ci95_high']):.3f}] \\\\")
    open(os.path.join(OUT, "coefficients.tex"), "w").write("\n".join(lines) + "\n")

    # Data description: correlations among inputs and residual sharing
    X, _ = load("3D_L")
    C = np.corrcoef(X.T)
    with open(os.path.join(OUT, "input_corr.txt"), "w") as fh:
        fh.write(f"r12={C[0,1]:.3f} r13={C[0,2]:.3f} r23={C[1,2]:.3f}\n")
    print("tables written to", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
