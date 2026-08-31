"""泄漏自检：打乱标签对照（CLAUDE.md 规则一的可执行版本）。

原理：把标签在**每个交易日的截面内**随机打乱，其余流程一字不改地重跑。
若管线干净，打乱后的 RankIC 应当在零附近（其分布即该管线的零假设分布）；
若显著偏离零，说明特征里含有当日截面的未来信息（前视），或评估流程本身有泄漏。

为什么在截面内打乱而不是全局打乱：全局打乱会同时破坏"日"这一层结构，
即使管线干净也可能得到非零结果（因为不同日的收益均值不同）。截面内打乱
只破坏"哪只股票对应哪个收益"，恰好是我们要检验的那一层对应关系。

用法::

    from crsp_pipeline.leak_check import shuffled_label_check
    res = shuffled_label_check(
        fit_predict=lambda tr, va: ridge(tr["X"], tr["y"], va["X"], alpha),
        train=dict(X=Xtr, y=ytr, day=dtr),
        valid=dict(X=Xva, y=yva, day=dva),
        n_draws=20)
    print(res["p_value"], res["null_mean"], res["actual"])

判读：``p_value`` 是打乱分布中 |IC| 不低于实际 |IC| 的比例。
- p < 0.05 且 null 分布居中于零 → 通过；
- null 分布**不**居中于零（|null_mean| 大于其标准差）→ **管线有泄漏，先修再谈结果**。
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _daily_rank_ic(pred: np.ndarray, y: np.ndarray, day: np.ndarray,
                   min_names: int = 50) -> float:
    out = []
    for u in np.unique(day):
        k = day == u
        if k.sum() < min_names:
            continue
        a = np.argsort(np.argsort(pred[k])).astype(np.float64)
        b = np.argsort(np.argsort(y[k])).astype(np.float64)
        out.append(np.corrcoef(a, b)[0, 1])
    return float(np.nanmean(out)) if out else float("nan")


def shuffle_within_day(y: np.ndarray, day: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """在每个交易日的截面内打乱标签，保留"日"这一层结构。"""
    out = y.copy()
    for u in np.unique(day):
        k = np.flatnonzero(day == u)
        out[k] = y[rng.permutation(k)]
    return out


def shuffled_label_check(
    fit_predict: Callable[[dict, dict], np.ndarray],
    train: dict,
    valid: dict,
    n_draws: int = 20,
    seed: int = 20260831,
    shuffle_train: bool = True,
    shuffle_valid: bool = True,
) -> dict:
    """跑 n_draws 次"标签打乱"对照，返回零分布与实际值的比较。

    Parameters
    ----------
    fit_predict
        ``f(train, valid) -> pred``。train/valid 是含 ``X``/``y``/``day`` 的 dict；
        函数内部只许用传进来的这两个 dict，不得再去读原始数据（否则打乱无效）。
    shuffle_train / shuffle_valid
        训练侧与评估侧是否都打乱。**两侧都打乱**检验的是"整条链路有没有泄漏"；
        只打乱评估侧检验的是"评估环节有没有泄漏"。默认两侧都打乱。
    """
    rng = np.random.default_rng(seed)
    actual = _daily_rank_ic(fit_predict(train, valid), valid["y"], valid["day"])

    null = []
    for _ in range(n_draws):
        tr = dict(train)
        va = dict(valid)
        if shuffle_train:
            tr["y"] = shuffle_within_day(train["y"], train["day"], rng)
        if shuffle_valid:
            va["y"] = shuffle_within_day(valid["y"], valid["day"], rng)
        null.append(_daily_rank_ic(fit_predict(tr, va), va["y"], va["day"]))
    null = np.asarray(null, dtype=float)

    nm, ns = float(np.nanmean(null)), float(np.nanstd(null))
    leaking = abs(nm) > max(ns, 1e-12)          # 零分布不居中于零 → 有泄漏
    return {
        "actual": actual,
        "null_mean": nm,
        "null_std": ns,
        "null_draws": null.tolist(),
        "p_value": float(np.mean(np.abs(null) >= abs(actual))),
        "leaking": bool(leaking),
        "verdict": ("零分布不居中于零 → 管线疑似泄漏，先修再谈结果" if leaking else
                    "通过：零分布居中于零" if np.mean(np.abs(null) >= abs(actual)) < 0.05 else
                    "零分布正常，但实际值未显著超出零分布（信号弱或无）"),
    }
