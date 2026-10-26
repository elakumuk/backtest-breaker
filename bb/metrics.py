"""Code-computed performance and backtest-overfitting metrics."""

from __future__ import annotations

from itertools import combinations
from math import erf, log, sqrt

import numpy as np
import pandas as pd


def sharpe(daily_net: pd.Series, rf: pd.Series, long_short: bool = False) -> float:
    """Annualized Sharpe (sqrt(252)); long-only uses excess returns, long-short raw.

    A constant positive series has zero estimated volatility, so it returns a
    large finite sentinel Sharpe rather than infinity.
    """
    values = daily_net.astype(float)
    if not long_short:
        values = values - rf.reindex(values.index).fillna(0.0)
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return float("nan")
    volatility = float(values.std(ddof=1))
    if volatility <= 1e-15:
        return 1_000_000.0 if float(values.mean()) > 0 else 0.0
    return float(values.mean() / volatility * sqrt(252))


def max_drawdown(daily_net: pd.Series) -> float:
    """Return the minimum peak-to-trough drawdown of a daily return stream."""
    equity = (1.0 + daily_net.fillna(0.0)).cumprod()
    return float((equity / equity.cummax() - 1.0).min()) if len(equity) else float("nan")


def annual_return(daily_net: pd.Series) -> float:
    """Geometrically annualize a daily net return series using 252 sessions."""
    values = daily_net.dropna().astype(float)
    if not len(values):
        return float("nan")
    terminal = float((1.0 + values).prod())
    return float(np.sign(terminal) * abs(terminal) ** (252.0 / len(values)) - 1.0)


def annual_turnover(turnover: pd.Series) -> float:
    """Average annual sum of one-way portfolio turnover."""
    if not len(turnover):
        return float("nan")
    return float(turnover.groupby(turnover.index.year).sum().mean())


def _normal_cdf(value: float) -> float:
    """Standard normal cumulative distribution without an extra dependency."""
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def deflated_sharpe_ratio(
    sr_hat: float,
    sr_trials: list[float] | np.ndarray,
    n_obs: int,
    skew: float,
    kurt: float,
) -> float:
    """Estimate the Deflated Sharpe Ratio probability (Bailey & Lopez de Prado, 2014).

    ``sr_hat`` and ``sr_trials`` are per-period, non-annualized Sharpe ratios;
    ``kurt`` is ordinary kurtosis (not excess kurtosis). The expected maximum
    null Sharpe is scaled by the sample standard deviation of trial Sharpe ratios.
    """
    trials = np.asarray(sr_trials, dtype=float)
    trials = trials[np.isfinite(trials)]
    if not len(trials) or n_obs < 2:
        return float("nan")
    count = len(trials)
    sigma_trials = float(np.std(trials, ddof=1)) if count > 1 else 0.0
    if count == 1:
        expected_max_unit = 0.0
    else:
        gamma = 0.5772156649015329
        q1 = _normal_quantile(1.0 - 1.0 / count)
        q2 = _normal_quantile(1.0 - 1.0 / (count * np.e))
        expected_max_unit = (1.0 - gamma) * q1 + gamma * q2
    sr_star = sigma_trials * expected_max_unit
    variance_term = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat**2
    if variance_term <= 0:
        return float(1.0 if sr_hat > sr_star else 0.0)
    standard_error = sqrt(variance_term / (n_obs - 1))
    return float(np.clip(_normal_cdf((sr_hat - sr_star) / standard_error), 0.0, 1.0))


def _normal_quantile(probability: float) -> float:
    """Approximate inverse standard normal CDF (Acklam rational approximation)."""
    p = min(max(probability, 1e-15), 1 - 1e-15)
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416]
    low, high = 0.02425, 1 - 0.02425
    if p < low:
        q = sqrt(-2 * log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > high:
        q = sqrt(-2 * log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def pbo_cscv(trial_returns: pd.DataFrame, n_splits: int = 16) -> tuple[float, list[float]]:
    """Compute CSCV Probability of Backtest Overfitting and logit distribution.

    Follows Bailey, Borwein, Lopez de Prado, and Zhu (2016). Returns are divided
    into an even number of contiguous blocks; every half-block combination is
    used in sample, and the selected trial is ranked on its complementary blocks.
    """
    if n_splits < 2 or n_splits % 2:
        raise ValueError("n_splits must be an even integer of at least 2")
    matrix = trial_returns.to_numpy(dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2 or matrix.shape[0] < n_splits:
        raise ValueError("trial_returns must contain at least n_splits rows and two trials")
    blocks = np.array_split(np.arange(matrix.shape[0]), n_splits)
    logits: list[float] = []
    half = n_splits // 2

    def trial_score(rows: np.ndarray) -> np.ndarray:
        sample = matrix[rows]
        means = np.nanmean(sample, axis=0)
        deviations = np.nanstd(sample, axis=0, ddof=1)
        return np.divide(means, deviations, out=np.full_like(means, np.nan), where=deviations > 0)

    for chosen in combinations(range(n_splits), half):
        in_rows = np.concatenate([blocks[i] for i in chosen])
        out_rows = np.concatenate([blocks[i] for i in range(n_splits) if i not in chosen])
        in_score = trial_score(in_rows)
        out_score = trial_score(out_rows)
        winner = int(np.nanargmax(in_score))
        rank = int(np.sum(out_score <= out_score[winner]))
        omega = np.clip(rank / (matrix.shape[1] + 1.0), 1e-12, 1 - 1e-12)
        logits.append(float(np.log(omega / (1 - omega))))
    return float(np.mean(np.asarray(logits) < 0)), logits
