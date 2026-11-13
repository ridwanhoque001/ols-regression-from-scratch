"""
verify.py
Checks the from-scratch OLS implementation against two independent solvers:
NumPy's general least squares routine (np.linalg.lstsq, SVD based) and
scikit-learn's LinearRegression. Both are freely available to all OU students.

CS 4033/5033 Machine Learning Fundamentals, Fall 2026, Project 2
Author: Muhammad Ridwanul Hoque

Output: ../results/verification.csv with the largest absolute difference in
coefficients and in R^2 for every dataset.
"""

import os
import csv
import numpy as np
from sklearn.linear_model import LinearRegression

from linreg import OLSRegression, r2_score
from experiment import load, DIMS, NOISE, RESULTS


def main():
    rows = []
    for d in DIMS:
        for nz in NOISE:
            name = f"{d}_{nz}"
            X, y = load(name)

            ours = OLSRegression().fit(X, y)

            A = np.column_stack([np.ones(len(y)), X])
            lstsq_coef = np.linalg.lstsq(A, y, rcond=None)[0]

            sk = LinearRegression().fit(X, y)
            sk_coef = np.concatenate([[sk.intercept_], sk.coef_])

            rows.append({
                "dataset": name,
                "max_coef_diff_vs_lstsq": float(np.max(np.abs(ours.coef_ - lstsq_coef))),
                "max_coef_diff_vs_sklearn": float(np.max(np.abs(ours.coef_ - sk_coef))),
                "R2_diff_vs_sklearn": abs(r2_score(y, ours.fitted_) - sk.score(X, y)),
            })
            print(rows[-1])

    with open(os.path.join(RESULTS, "verification.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
