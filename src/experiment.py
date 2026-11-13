"""
experiment.py
Fits one OLS model to each of the nine datasets and records every metric.

CS 4033/5033 Machine Learning Fundamentals, Fall 2026, Project 2
Author: Muhammad Ridwanul Hoque

Outputs (all in ../results):
    summary.csv           one row per dataset: fit and generalization metrics
    coefficients.csv      one row per coefficient: estimate, SE, t, p, 95% CI
    equations.txt         the fitted equation for each dataset
    predictions/<name>.csv  inputs, actual y, fitted y and residual per item
"""

import os
import csv
import numpy as np

from linreg import OLSRegression, rmse, mae, r2_score, adjusted_r2, kfold_cv

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RESULTS = os.path.join(HERE, "..", "results")

DIMS = ["1D", "2D", "3D"]
NOISE = ["L", "M", "H"]
NOISE_NAME = {"L": "Low", "M": "Moderate", "H": "High"}


def load(name):
    """Load one dataset. Returns inputs X (n x p) and output y (n)."""
    a = np.loadtxt(os.path.join(DATA, name + ".csv"), delimiter=",", skiprows=1)
    return a[:, :-1], a[:, -1]


def equation_string(coef):
    """Format y = b0 + b1 x1 + ... with four decimals."""
    terms = [f"{coef[0]:.4f}"]
    names = ["x"] if len(coef) == 2 else [f"x{i}" for i in range(1, len(coef))]
    for b, nm in zip(coef[1:], names):
        sign = "-" if b < 0 else "+"
        terms.append(f"{sign} {abs(b):.4f} {nm}")
    return "y = " + " ".join(terms)


def main():
    os.makedirs(os.path.join(RESULTS, "predictions"), exist_ok=True)
    summary_rows, coef_rows, eq_lines = [], [], []

    for d in DIMS:
        for nz in NOISE:
            name = f"{d}_{nz}"
            X, y = load(name)
            n, p = X.shape

            model = OLSRegression().fit(X, y)
            yhat = model.fitted_
            r2 = r2_score(y, yhat)
            F, Fp = model.f_test()
            sw_stat, sw_p = model.shapiro()
            bp_stat, bp_p = model.breusch_pagan()
            cv_rmse, cv_rmse_sd, cv_r2, cv_r2_sd = kfold_cv(X, y, k=10, repeats=10, seed=0)

            summary_rows.append({
                "dataset": name, "dims": p, "noise": NOISE_NAME[nz], "n": n,
                "R2": r2, "adj_R2": adjusted_r2(r2, n, p),
                "RMSE": rmse(y, yhat), "MAE": mae(y, yhat),
                "sigma_hat": model.sigma_hat, "y_std": float(np.std(y)),
                "F": F, "F_p": Fp,
                "CV_RMSE": cv_rmse, "CV_RMSE_sd": cv_rmse_sd,
                "CV_R2": cv_r2, "CV_R2_sd": cv_r2_sd,
                "shapiro_W": sw_stat, "shapiro_p": sw_p,
                "BP_LM": bp_stat, "BP_p": bp_p,
            })

            tab = model.coef_table()
            labels = ["b0 (intercept)"] + [f"b{j} (x{j})" for j in range(1, p + 1)]
            for j, lab in enumerate(labels):
                coef_rows.append({
                    "dataset": name, "term": lab,
                    "estimate": tab["coef"][j], "se": tab["se"][j],
                    "t": tab["t"][j], "p": tab["p"][j],
                    "ci95_low": tab["ci_low"][j], "ci95_high": tab["ci_high"][j],
                })

            eq_lines.append(f"{name:5s}  {equation_string(model.coef_)}")

            # Per-item predictions, for plotting and for the TA to check
            cols = [f"x{j}" for j in range(1, p + 1)]
            with open(os.path.join(RESULTS, "predictions", name + ".csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(cols + ["y", "y_hat", "residual"])
                for xi, yi, fi in zip(X, y, yhat):
                    w.writerow([f"{v:.4f}" for v in xi] + [f"{yi:.4f}", f"{fi:.6f}", f"{yi - fi:.6f}"])

            print(f"{name}: R2={r2:.3f}  RMSE={rmse(y, yhat):.4f}  CV_RMSE={cv_rmse:.4f}  "
                  f"{equation_string(model.coef_)}")

    def write(path, rows):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    write(os.path.join(RESULTS, "summary.csv"), summary_rows)
    write(os.path.join(RESULTS, "coefficients.csv"), coef_rows)
    with open(os.path.join(RESULTS, "equations.txt"), "w") as fh:
        fh.write("\n".join(eq_lines) + "\n")


if __name__ == "__main__":
    main()
