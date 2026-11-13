"""
plots.py
Every figure in the Project 2 report.

CS 4033/5033 Machine Learning Fundamentals, Fall 2026, Project 2
Author: Muhammad Ridwanul Hoque

Run after experiment.py. Figures are written to ../figures.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from linreg import OLSRegression
from experiment import load, DIMS, NOISE, NOISE_NAME, RESULTS

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
COL = {"L": "#1f77b4", "M": "#ff7f0e", "H": "#d62728"}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=170, bbox_inches="tight")
    plt.close(fig)


def read_summary():
    with open(os.path.join(RESULTS, "summary.csv")) as fh:
        return {r["dataset"]: r for r in csv.DictReader(fh)}


def read_coefs():
    with open(os.path.join(RESULTS, "coefficients.csv")) as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------- #
# Figures that describe the data
# ---------------------------------------------------------------------- #
def fig_data_1d():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, nz in zip(axes, NOISE):
        X, y = load(f"1D_{nz}")
        ax.scatter(X[:, 0], y, s=16, color=COL[nz])
        ax.set_title(f"1D {NOISE_NAME[nz]} noise (r = {np.corrcoef(X[:, 0], y)[0, 1]:.2f})")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    save(fig, "fig1_data_1d.png")


def fig_data_2d():
    fig = plt.figure(figsize=(14, 4.4))
    for i, nz in enumerate(NOISE):
        X, y = load(f"2D_{nz}")
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(X[:, 0], X[:, 1], y, s=12, color=COL[nz])
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("y")
        ax.set_title(f"2D {NOISE_NAME[nz]} noise")
        ax.view_init(elev=20, azim=-60)
    save(fig, "fig2_data_2d.png")


def fig_data_3d():
    fig, axes = plt.subplots(3, 3, figsize=(12, 9.5), sharey="row")
    for i, nz in enumerate(NOISE):
        X, y = load(f"3D_{nz}")
        for j in range(3):
            ax = axes[i, j]
            ax.scatter(X[:, j], y, s=10, color=COL[nz])
            r = np.corrcoef(X[:, j], y)[0, 1]
            ax.set_title(f"3D {NOISE_NAME[nz]}: y vs x{j + 1} (r = {r:.2f})", fontsize=9)
            if i == 2:
                ax.set_xlabel(f"x{j + 1}")
        axes[i, 0].set_ylabel("y")
    fig.tight_layout()
    save(fig, "fig3_data_3d.png")


# ---------------------------------------------------------------------- #
# Figures that show the fitted models
# ---------------------------------------------------------------------- #
def fig_fit_1d():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, nz in zip(axes, NOISE):
        X, y = load(f"1D_{nz}")
        m = OLSRegression().fit(X, y)
        # residual segments
        for xi, yi, fi in zip(X[:, 0], y, m.fitted_):
            ax.plot([xi, xi], [yi, fi], color="0.6", lw=0.7, zorder=1)
        ax.scatter(X[:, 0], y, s=16, color=COL[nz], zorder=3, label="data")
        # line and 95% confidence band for the mean response
        g = np.linspace(0, 1, 200)
        A = np.column_stack([np.ones_like(g), g])
        cov = m.sigma_hat ** 2 * (m._R_inv @ m._R_inv.T)
        se = np.sqrt(np.einsum("ij,jk,ik->i", A, cov, A))
        t = stats.t.ppf(0.975, m.df_resid)
        fit = A @ m.coef_
        ax.fill_between(g, fit - t * se, fit + t * se, color="k", alpha=0.15, label="95% CI of mean")
        ax.plot(g, fit, "k-", lw=2, label=f"$\\hat{{y}} = {m.coef_[0]:.3f} + {m.coef_[1]:.3f}x$")
        ax.set_title(f"1D {NOISE_NAME[nz]} noise")
        ax.set_xlabel("x")
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("y")
    save(fig, "fig4_fit_1d.png")


def fig_fit_2d():
    fig = plt.figure(figsize=(15, 4.8))
    for i, nz in enumerate(NOISE):
        X, y = load(f"2D_{nz}")
        m = OLSRegression().fit(X, y)
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        g1, g2 = np.meshgrid(np.linspace(0, 1, 15), np.linspace(0, 1, 15))
        Z = m.coef_[0] + m.coef_[1] * g1 + m.coef_[2] * g2
        ax.plot_surface(g1, g2, Z, alpha=0.35, color="0.5", edgecolor="0.4", lw=0.3)
        above = m.resid_ >= 0
        for xi, yi, fi in zip(X, y, m.fitted_):
            ax.plot([xi[0], xi[0]], [xi[1], xi[1]], [yi, fi], color="0.3", lw=0.6)
        ax.scatter(X[above, 0], X[above, 1], y[above], s=12, color=COL[nz], depthshade=False, label="above plane")
        ax.scatter(X[~above, 0], X[~above, 1], y[~above], s=12, facecolors="none",
                   edgecolors=COL[nz], depthshade=False, label="below plane")
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("y")
        ax.set_title(f"2D {NOISE_NAME[nz]}: $\\hat{{y}} = {m.coef_[0]:.3f} {m.coef_[1]:+.3f}x_1 {m.coef_[2]:+.3f}x_2$",
                     fontsize=9)
        ax.view_init(elev=18, azim=-55)
        ax.legend(fontsize=7, loc="upper left")
    save(fig, "fig5_fit_2d.png")


def fig_pred_vs_actual():
    fig, axes = plt.subplots(3, 3, figsize=(11, 10.5))
    for i, d in enumerate(DIMS):
        for j, nz in enumerate(NOISE):
            X, y = load(f"{d}_{nz}")
            m = OLSRegression().fit(X, y)
            ax = axes[i, j]
            lo, hi = min(y.min(), m.fitted_.min()) - 0.03, max(y.max(), m.fitted_.max()) + 0.03
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect prediction")
            ax.scatter(y, m.fitted_, s=12, color=COL[nz])
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
            r2 = 1 - (m.resid_ @ m.resid_) / ((y - y.mean()) @ (y - y.mean()))
            ax.set_title(f"{d} {NOISE_NAME[nz]} (R$^2$ = {r2:.3f})", fontsize=9)
            if i == 2:
                ax.set_xlabel("actual y")
            if j == 0:
                ax.set_ylabel("predicted y")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fig6_pred_vs_actual.png")


def fig_residuals():
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=False, sharey=True)
    for i, d in enumerate(DIMS):
        for j, nz in enumerate(NOISE):
            X, y = load(f"{d}_{nz}")
            m = OLSRegression().fit(X, y)
            ax = axes[i, j]
            ax.axhline(0, color="k", lw=1)
            s = m.sigma_hat
            ax.axhspan(-2 * s, 2 * s, color=COL[nz], alpha=0.08)
            ax.scatter(m.fitted_, m.resid_, s=12, color=COL[nz])
            ax.set_title(f"{d} {NOISE_NAME[nz]} (band = $\\pm 2\\hat\\sigma$)", fontsize=9)
            if i == 2:
                ax.set_xlabel("fitted y")
            if j == 0:
                ax.set_ylabel("residual")
    fig.tight_layout()
    save(fig, "fig7_residuals.png")


def fig_qq():
    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    summ = read_summary()
    for i, d in enumerate(DIMS):
        for j, nz in enumerate(NOISE):
            X, y = load(f"{d}_{nz}")
            m = OLSRegression().fit(X, y)
            ax = axes[i, j]
            (osm, osr), (slope, icpt, _) = stats.probplot(m.resid_, dist="norm")
            ax.scatter(osm, osr, s=10, color=COL[nz])
            ax.plot(osm, slope * osm + icpt, "k-", lw=1)
            p = float(summ[f"{d}_{nz}"]["shapiro_p"])
            ax.set_title(f"{d} {NOISE_NAME[nz]} (Shapiro-Wilk p = {p:.2f})", fontsize=9)
            if i == 2:
                ax.set_xlabel("theoretical normal quantile")
            if j == 0:
                ax.set_ylabel("residual quantile")
    fig.tight_layout()
    save(fig, "fig8_qq.png")


# ---------------------------------------------------------------------- #
# Figures that compare the nine models
# ---------------------------------------------------------------------- #
def fig_metrics():
    summ = read_summary()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(3)
    w = 0.26
    for k, nz in enumerate(NOISE):
        rows = [summ[f"{d}_{nz}"] for d in DIMS]
        off = (k - 1) * w
        axes[0].bar(x + off, [float(r["R2"]) for r in rows], w, color=COL[nz], label=f"{NOISE_NAME[nz]} noise")
        axes[0].plot(x + off, [float(r["CV_R2"]) for r in rows], "k_", ms=14, mew=2)
        axes[1].bar(x + off, [float(r["RMSE"]) for r in rows], w, color=COL[nz])
        axes[1].plot(x + off, [float(r["CV_RMSE"]) for r in rows], "k_", ms=14, mew=2)
        axes[2].plot(x, [float(r["adj_R2"]) for r in rows], "o-", color=COL[nz], label=f"{NOISE_NAME[nz]}")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(["1 predictor", "2 predictors", "3 predictors"])
    axes[0].set_ylabel("R$^2$ (bar) and 10-fold CV R$^2$ (black tick)")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].set_ylabel("RMSE (bar) and 10-fold CV RMSE (black tick)")
    axes[2].set_ylabel("adjusted R$^2$")
    axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=8)
    axes[0].set_title("Share of variance explained")
    axes[1].set_title("Typical prediction error (units of y)")
    axes[2].set_title("Adjusted R$^2$ against number of predictors")
    fig.tight_layout()
    save(fig, "fig9_metrics.png")


def fig_coefs():
    rows = read_coefs()
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    terms = ["b0 (intercept)", "b1 (x1)", "b2 (x2)", "b3 (x3)"]
    labels = [f"{d} {nz}" for d in DIMS for nz in NOISE]
    for ax, term in zip(axes, terms):
        for k, lab in enumerate(labels):
            d, nz = lab.split()
            r = [q for q in rows if q["dataset"] == f"{d}_{nz}" and q["term"] == term]
            if not r:
                continue
            r = r[0]
            est, lo, hi = float(r["estimate"]), float(r["ci95_low"]), float(r["ci95_high"])
            ax.errorbar(est, k, xerr=[[est - lo], [hi - est]], fmt="o", color=COL[nz], capsize=3)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(term)
        ax.set_xlabel("estimate with 95% CI")
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, "fig10_coefficients.png")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_data_1d(); fig_data_2d(); fig_data_3d()
    fig_fit_1d(); fig_fit_2d(); fig_pred_vs_actual()
    fig_residuals(); fig_qq(); fig_metrics(); fig_coefs()
    print("figures written to", os.path.abspath(FIG))
