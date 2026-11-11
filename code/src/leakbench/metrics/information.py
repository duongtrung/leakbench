"""Information-theoretic quantities: entropy, confusion-matrix MI, g-leakage.

These implement equations E4/E5/E7 of the spec. Numerically safe: zero cells
contribute 0 (``0*log0 := 0``), empty inputs return 0, degenerate priors are
guarded.
"""

from __future__ import annotations

import numpy as np


def entropy_bits(prob: np.ndarray) -> float:
    """Shannon entropy in bits of a probability vector."""
    p = np.asarray(prob, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def confusion_mutual_information(confusion: np.ndarray) -> float:
    """``I(C; Ĉ) = sum_ij p_ij log2( p_ij / (p_i. p_.j) )`` (Def. "MI lower bound").

    Parameters
    ----------
    confusion:
        raw counts, rows = true class, cols = predicted class.

    Returns
    -------
    Mutual information in bits between true and predicted labels. This is a
    **lower bound** on ``I(C; O_v)`` for any classifier (data-processing). Never an
    upper bound on privacy.
    """
    M = np.asarray(confusion, dtype=float)
    total = M.sum()
    if total <= 0:
        return 0.0
    P = M / total                      # joint p_ij
    p_row = P.sum(axis=1, keepdims=True)   # p_i.
    p_col = P.sum(axis=0, keepdims=True)   # p_.j
    denom = p_row @ p_col
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((P > 0) & (denom > 0), P / denom, 1.0)
        terms = np.where((P > 0) & (denom > 0), P * np.log2(ratio), 0.0)
    return float(terms.sum())


def identity_gain_matrix(num_classes: int) -> np.ndarray:
    """Identity gain ``g(t, ĥt) = 1[t == ĥt]`` --- recovers the advantage (Def. g-leakage)."""
    return np.eye(num_classes, dtype=float)


def family_gain_matrix(class_families: np.ndarray) -> np.ndarray:
    """Family gain ``g(t, ĥt) = 1[family(t) == family(ĥt)]`` --- scores coarse-secret
    (family-level) guessing without changing the estimator (Def. g-leakage view)."""
    fam = np.asarray(class_families)
    K = fam.shape[0]
    G = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            G[i, j] = 1.0 if fam[i] == fam[j] else 0.0
    return G


def g_leakage(
    prior: np.ndarray,
    posterior_joint: np.ndarray,
    gain: np.ndarray,
) -> dict[str, float]:
    """Multiplicative g-leakage ``L_g = V_g(pi, O) / V_g(pi)`` (Def. "g-leakage view").

    Parameters
    ----------
    prior:
        prior ``pi`` over classes, shape (K,).
    posterior_joint:
        joint distribution ``P(t, x)`` approximated by the normalized confusion
        matrix rows=true, cols=observed-bucket (here, predicted class), shape (K, M).
    gain:
        gain matrix ``g(t, ĥt)``, shape (K, K_guess). For identity/family, K_guess=K.

    Returns
    -------
    dict with prior gain ``V_prior``, posterior gain ``V_post``, multiplicative
    ``L_mult`` and additive ``L_add`` leakage.
    """
    pi = np.asarray(prior, dtype=float)
    P = np.asarray(posterior_joint, dtype=float)
    G = np.asarray(gain, dtype=float)

    # Prior-optimal guess: max over guesses of sum_t pi(t) g(t, guess).
    V_prior = float(np.max(pi @ G))

    total = P.sum()
    if total <= 0:
        return {"V_prior": V_prior, "V_post": V_prior, "L_mult": 1.0, "L_add": 0.0}
    P = P / total
    p_x = P.sum(axis=0)                 # marginal over observed buckets
    V_post = 0.0
    for x in range(P.shape[1]):
        if p_x[x] <= 0:
            continue
        post_t = P[:, x] / p_x[x]       # Pr(t | x)
        best = float(np.max(post_t @ G))
        V_post += p_x[x] * best
    L_mult = V_post / V_prior if V_prior > 0 else float("inf")
    return {
        "V_prior": V_prior,
        "V_post": float(V_post),
        "L_mult": float(L_mult),
        "L_add": float(V_post - V_prior),
    }
