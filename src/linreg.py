"""
linreg.py
Ordinary least squares (OLS) linear regression, written from scratch in NumPy.

CS 4033/5033 Machine Learning Fundamentals, Fall 2026, Project 2
Author: Muhammad Ridwanul Hoque

The model is
    y = b0 + b1*x1 + ... + bp*xp + e,   e ~ N(0, sigma^2)

Coefficients are found by minimizing the residual sum of squares (RSS).
Instead of inverting X^T X directly (the textbook normal equations), the
solver uses a QR decomposition of the design matrix, X = QR, and solves the
triangular system R b = Q^T y. This gives the same answer as the normal
equations but is numerically more stable when columns are close to collinear.

SciPy is used only for the Student t, F and chi-square distribution
functions that turn test statistics into p-values. Everything else (the fit,
standard errors, confidence intervals, R^2, RMSE, the F test, the
Breusch-Pagan test) is computed here.
"""

import numpy as np
from scipy import stats


class OLSRegression:
    """Multiple linear regression fitted by ordinary least squares."""

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None          # full coefficient vector [b0, b1, ..., bp]
        self.n_ = None             # number of data items
        self.p_ = None             # number of predictors (not counting b0)

    # ------------------------------------------------------------------ #
    # Fitting and prediction
    # ------------------------------------------------------------------ #
    def _design(self, X):
        """Return the design matrix: a column of ones followed by X."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X

    def fit(self, X, y):
        """Estimate the coefficients from inputs X (n x p) and outputs y (n)."""
        A = self._design(X)
        y = np.asarray(y, dtype=float)
        self.n_, k = A.shape
        self.p_ = k - 1 if self.fit_intercept else k

        # QR solve of the least squares problem min ||A b - y||^2
        Q, R = np.linalg.qr(A)              # reduced QR: Q is n x k, R is k x k
        self.coef_ = np.linalg.solve(R, Q.T @ y)

        # Quantities needed for inference
        self._A = A
        self._y = y
        self._R_inv = np.linalg.inv(R)       # (A^T A)^-1 = R^-1 R^-T
        self.fitted_ = A @ self.coef_
        self.resid_ = y - self.fitted_
        return self

    def predict(self, X):
        """Predicted outputs for new inputs X."""
        return self._design(X) @ self.coef_

    # ------------------------------------------------------------------ #
    # Inference on the coefficients
    # ------------------------------------------------------------------ #
    @property
    def df_resid(self):
        """Residual degrees of freedom, n - p - 1."""
        return self.n_ - self._A.shape[1]

    @property
    def sigma_hat(self):
        """Residual standard error, an unbiased estimate of the noise sigma."""
        rss = float(self.resid_ @ self.resid_)
        return np.sqrt(rss / self.df_resid)

    def coef_table(self, alpha=0.05):
        """Standard errors, t statistics, p-values and (1 - alpha) CIs."""
        cov = self.sigma_hat ** 2 * (self._R_inv @ self._R_inv.T)
        se = np.sqrt(np.diag(cov))
        t = self.coef_ / se
        p = 2.0 * stats.t.sf(np.abs(t), self.df_resid)
        tcrit = stats.t.ppf(1.0 - alpha / 2.0, self.df_resid)
        return {
            "coef": self.coef_,
            "se": se,
            "t": t,
            "p": p,
            "ci_low": self.coef_ - tcrit * se,
            "ci_high": self.coef_ + tcrit * se,
        }

    def f_test(self):
        """Overall F test of H0: b1 = ... = bp = 0 (intercept-only model)."""
        y = self._y
        tss = float(((y - y.mean()) ** 2).sum())
        rss = float(self.resid_ @ self.resid_)
        F = ((tss - rss) / self.p_) / (rss / self.df_resid)
        p = stats.f.sf(F, self.p_, self.df_resid)
        return F, p

    def breusch_pagan(self):
        """Breusch-Pagan test for heteroscedasticity (Koenker's version).

        Regress the squared residuals on the predictors; under constant
        variance, LM = n * R^2 of that auxiliary regression ~ chi^2(p).
        """
        e2 = self.resid_ ** 2
        aux = OLSRegression().fit(self._A[:, 1:], e2)
        lm = self.n_ * r2_score(e2, aux.fitted_)
        return lm, stats.chi2.sf(lm, self.p_)

    def shapiro(self):
        """Shapiro-Wilk test for normality of the residuals."""
        return stats.shapiro(self.resid_)


# ---------------------------------------------------------------------- #
# Performance metrics
# ---------------------------------------------------------------------- #
def rss(y, yhat):
    """Residual sum of squares."""
    r = np.asarray(y) - np.asarray(yhat)
    return float(r @ r)


def rmse(y, yhat):
    """Root mean squared error, in the units of y."""
    return float(np.sqrt(rss(y, yhat) / len(y)))


def mae(y, yhat):
    """Mean absolute error, in the units of y."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def r2_score(y, yhat):
    """Coefficient of determination, 1 - RSS / TSS."""
    y = np.asarray(y)
    tss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - rss(y, yhat) / tss


def adjusted_r2(r2, n, p):
    """R^2 penalized for the number of predictors p."""
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)


def kfold_cv(X, y, k=10, repeats=10, seed=0):
    """Repeated k-fold cross-validation.

    Returns the mean and standard deviation (over repeats) of the
    out-of-fold RMSE and out-of-fold R^2. Each repeat shuffles the data
    with a different seed, so the result does not depend on one split.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    rmses, r2s = [], []
    for _ in range(repeats):
        order = rng.permutation(n)
        folds = np.array_split(order, k)
        pred = np.empty(n)
        for f in folds:
            train = np.setdiff1d(order, f)
            model = OLSRegression().fit(X[train], y[train])
            pred[f] = model.predict(X[f])
        rmses.append(rmse(y, pred))
        r2s.append(r2_score(y, pred))
    return float(np.mean(rmses)), float(np.std(rmses)), float(np.mean(r2s)), float(np.std(r2s))
