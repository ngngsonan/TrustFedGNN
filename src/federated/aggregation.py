"""Server-side aggregation rules.

All rules operate on a list of client *pseudo-gradients* ``g_k = theta_global -
theta_local_k`` (flat tensors). The server applies ``theta_new = theta_global -
g_agg``; with weights summing to 1 this reduces to a weighted average of local
models, so FedAvg, BFWA and the robust rules share one interface.

Rules
    fedavg        weighted by client sample count.
    bfwa          Bi-objective Frank-Wolfe: maximise weighted utility s.t. a
                  hard bound tau on weighted demographic-parity gap (ours).
    krum          Blanchard et al. 2017 -- pick the update closest to its
                  n-f-2 nearest neighbours.
    multikrum     average the m best Krum-scored updates.
    median        coordinate-wise median (Yin et al. 2018).
    trimmed_mean  coordinate-wise beta-trimmed mean (Yin et al. 2018).
    robust_bfwa   distance-screen Byzantine updates, then BFWA on survivors
                  with the fairness constraint (ours; robust + fair).
    fairgfl       weight ~ 1/(1+heterogeneity); heterogeneity proxied by a
                  client's sample-count deviation from the mean (Khan-style
                  overlap-ratio reweighting; see docs/BASELINES_AND_SOURCES.md).
    fedgraphfair  minimax/DRO dual-ascent reweighting toward high-loss clients,
                  simplex-projected, with lambda persisted across rounds.
    popets_fairfed FHE-friendly FairFed: degree-2 polynomial in place of
                  exp(-beta|.|) so the weighting stays homomorphically
                  computable (the crypto itself is not reimplemented).
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

import torch


# --------------------------------------------------------------------------- #
# Frank-Wolfe fairness-constrained weights
# --------------------------------------------------------------------------- #
def _scale(v: torch.Tensor) -> float:
    """Positive scale of a K-vector: its range, falling back to its magnitude
    and finally to 1.0 (K=1 or all-equal entries)."""
    if v.numel() == 0:
        return 1.0
    rng = float(v.max() - v.min())
    if rng > 1e-12:
        return rng
    mag = float(v.abs().max())
    return mag if mag > 1e-12 else 1.0


def bfwa_weights(perf: torch.Tensor, dpd: torch.Tensor, tau: float,
                 iters: int = 20, dual_step: float = 0.1,
                 w_min_factor: float = 0.2, mu_init: float = 0.0,
                 info: Dict = None) -> torch.Tensor:
    """Solve  max_w  w.perf  s.t.  w.dpd <= tau,  w in simplex  via Frank-Wolfe
    with a dual variable enforcing the fairness budget.

    Args:
        mu_init: starting value of the dual multiplier. The caller passes the
            multiplier carried over from the previous communication round when
            ``bfwa_persist_dual`` is on (see :func:`aggregate`); ``0.0``
            restarts the dual ascent every call.
        info: optional dict filled in-place with ``bfwa_mu`` (the multiplier
            after this call, to be persisted), the constraint residual
            ``w.dpd - tau`` and the feasibility flag, both BEFORE
            (``*_preclamp``) and AFTER the weight-floor clamp.

    Two properties this loop must have, and did not before (both are pinned by
    ``tests/test_revision_invariants.py``):

    * **The Frank-Wolfe averaging must actually average.** ``gamma = 2/(t+2)``
      evaluated at ``t = 0`` is ``1.0``: the first step *discards* the uniform
      iterate and jumps straight onto the vertex ``argmin(-perf + mu*dpd)``.
      Because the objective is linear in ``w`` its gradient does not depend on
      ``w``, so every later step re-selects that same vertex until ``mu`` grows
      large enough to flip it -- which, at the shipped ``iters=20``, never
      happened, and ``tau`` had literally no effect on the returned weights.
      The classic schedule is therefore run from ``t = 1`` (``gamma_1 = 2/3``),
      so the uniform initialisation is mixed into the iterate rather than
      erased.
    * **The dual must be able to reach the binding regime.** The vertex flips
      only once ``mu`` exceeds (utility gap)/(disparity gap), a ratio in the
      units of ``perf`` per unit of ``dpd``. With raw units and the shipped
      ``dual_step=0.1`` that takes hundreds of iterations. The Lagrangian is
      therefore evaluated on range-normalised terms, which makes ``mu``
      dimensionless (order 1 at the flip point) and gives ``dual_step`` the
      same meaning across datasets, scales and rounds -- necessary for the
      multiplier to be persisted across rounds at all.
    """
    K = perf.numel()
    dev = perf.device
    w = torch.full((K,), 1.0 / K, device=dev)
    mu = float(mu_init)
    perf_scale = _scale(perf)                       # dimensionless Lagrangian
    dpd_scale = _scale(dpd)
    for t in range(1, iters + 1):                   # t=1 -> gamma=2/3, not 1.0
        violation = float(torch.dot(w, dpd) - tau)
        grad = -perf / perf_scale + mu * dpd / dpd_scale   # d/dw of Lagrangian
        s = torch.zeros(K, device=dev)
        s[int(torch.argmin(grad))] = 1.0            # LMO on simplex
        gamma = 2.0 / (t + 2.0)                     # decaying step (spec-compliant)
        w = (1 - gamma) * w + gamma * s
        mu = max(0.0, mu + dual_step * violation / dpd_scale)
    resid_pre = float(torch.dot(w, dpd)) - tau
    # weight floor + renormalise (avoid a single client dominating)
    w = torch.clamp(w, min=w_min_factor / K)
    w = w / w.sum()
    resid_post = float(torch.dot(w, dpd)) - tau
    if info is not None:
        # The floor is part of the shipped rule, so the headline residual is the
        # post-clamp one -- it is what the aggregated model actually incurs. The
        # pre-clamp value is reported alongside it so the floor's cost in
        # feasibility is visible rather than hidden.
        info["bfwa_mu"] = mu
        info["constraint_residual_preclamp"] = resid_pre
        info["feasible_preclamp"] = bool(resid_pre <= 0.0)
        info["constraint_residual"] = resid_post
        info["feasible"] = bool(resid_post <= 0.0)
    return w


# --------------------------------------------------------------------------- #
# Robust primitives
# --------------------------------------------------------------------------- #
def _pairwise_sq_dists(mat: torch.Tensor) -> torch.Tensor:
    return torch.cdist(mat, mat) ** 2


def krum_scores(updates: List[torch.Tensor], f: int) -> torch.Tensor:
    K = len(updates)
    mat = torch.stack([u.flatten() for u in updates])
    d = _pairwise_sq_dists(mat)
    m = max(1, K - f - 2)                            # neighbours to sum
    scores = torch.empty(K, device=mat.device)
    for i in range(K):
        di = torch.cat([d[i, :i], d[i, i + 1:]])
        scores[i] = torch.sort(di).values[:m].sum()
    return scores


# --------------------------------------------------------------------------- #
# Null players
# --------------------------------------------------------------------------- #
def _zero_null_players(w: torch.Tensor, stack: torch.Tensor) -> torch.Tensor:
    """Force weight 0 on clients whose update is exactly zero, then renormalise.

    Null-player axiom: a client that submitted nothing must receive no credit.
    Any *fallback* that spreads weight uniformly (or by sample size) over all K
    clients silently breaks it, so the mask is applied to every weight vector
    the FU-Shapley branch can produce, not just to the ReLU-gated one.
    """
    null = stack.abs().amax(dim=1) == 0
    if not bool(null.any()):
        return w
    w = w.clone()
    w[null] = 0.0
    total = float(w.sum())
    if total > 0:
        return w / total
    active = (~null).to(w.dtype)
    if float(active.sum()) > 0:
        return active / active.sum()
    return torch.full_like(w, 1.0 / w.numel())   # everyone null: aggregate is 0


# --------------------------------------------------------------------------- #
# Cross-round dual state for BFWA
# --------------------------------------------------------------------------- #
def _read_mu(state: Dict, key: str, persist: bool) -> float:
    """Dual multiplier carried in from the previous round (0.0 if none)."""
    if not persist or state is None:
        return 0.0
    return float(state.get(key, 0.0))


def _write_mu(state: Dict, key: str, mu: float, persist: bool) -> None:
    if persist and state is not None:
        state[key] = float(mu)


# --------------------------------------------------------------------------- #
# GuardFed-style holdout measurement (ADR-27, K7)
# --------------------------------------------------------------------------- #
def compute_guardfed_holdout(
    model, global_flat: torch.Tensor, updates: List[torch.Tensor],
    data_holdout, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute g_s (reweighted task gradient) and Fair_k (EO gap of theta - Delta_k) on holdout.

    1. g_s: task gradient on holdout with reweighting w_i = P(a)*P(y)/P(a,y).
       Same sign convention as g_target of FU.
    2. Fair_k: EO gap (Eq. 12 of GuardFed) of theta - Delta_k on holdout,
       using hard prediction at 0.5 threshold.
    """
    from .client import load_flat_state
    from ..trust.incentive import _flat_grad_like_state

    model.eval()
    x = data_holdout.x
    ei = data_holdout.edge_index
    s = data_holdout.sensitive_attr
    m = data_holdout.val_mask
    y = data_holdout.y
    s_m, y_m = s[m], y[m]

    # --- 1. Compute g_s ---
    load_flat_state(model, global_flat.to(device))
    # Marginals
    p0 = (s_m == 0).float().mean()
    p1 = (s_m == 1).float().mean()
    py0 = (y_m == 0).float().mean()
    py1 = (y_m == 1).float().mean()
    p_a = torch.where(s_m == 1, p1, p0)
    p_y = torch.where(y_m == 1, py1, py0)

    # Joint
    p00 = ((s_m == 0) & (y_m == 0)).float().mean()
    p01 = ((s_m == 0) & (y_m == 1)).float().mean()
    p10 = ((s_m == 1) & (y_m == 0)).float().mean()
    p11 = ((s_m == 1) & (y_m == 1)).float().mean()

    p_ay = torch.zeros_like(p_a)
    p_ay[(s_m == 0) & (y_m == 0)] = p00
    p_ay[(s_m == 0) & (y_m == 1)] = p01
    p_ay[(s_m == 1) & (y_m == 0)] = p10
    p_ay[(s_m == 1) & (y_m == 1)] = p11

    # Sample weights w_i = P(a)*P(y)/P(a, y)
    w_i = torch.where(p_ay > 1e-12, (p_a * p_y) / (p_ay + 1e-12), torch.ones_like(p_a))

    model.zero_grad(set_to_none=True)
    pred = model(x, ei, s)[m]
    # Reweighted BCE loss
    bce = -(y_m.float() * torch.log(pred + 1e-7) + (1.0 - y_m.float()) * torch.log(1.0 - pred + 1e-7))
    loss_reweighted = (w_i * bce).mean()
    loss_reweighted.backward()
    g_s = _flat_grad_like_state(model).cpu()
    model.zero_grad(set_to_none=True)

    # --- 2. Compute Fair_k for each client candidate model theta - Delta_k ---
    K = len(updates)
    fair_list = []
    mask_y1 = (y_m == 1)
    mask_s0_y1 = mask_y1 & (s_m == 0)
    mask_s1_y1 = mask_y1 & (s_m == 1)

    for k in range(K):
        cand_flat = global_flat - updates[k].cpu()
        load_flat_state(model, cand_flat.to(device))
        with torch.no_grad():
            cand_pred = model(x, ei, s)[m]
            cand_hard = (cand_pred >= 0.5).float()
            if mask_s0_y1.any() and mask_s1_y1.any():
                tpr0 = cand_hard[mask_s0_y1].mean()
                tpr1 = cand_hard[mask_s1_y1].mean()
                fair_k = float(torch.abs(tpr0 - tpr1).item())
            else:
                fair_k = 0.0
        fair_list.append(fair_k)

    # Restore model to global_flat
    load_flat_state(model, global_flat.to(device))
    gs_fair = torch.tensor(fair_list, dtype=torch.float32)

    return g_s, gs_fair


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def aggregate(method: str, updates: List[torch.Tensor], meta: List[dict],
              *, tau: float = 0.05, fw_iters: int = 20, dual_step: float = 0.1,
              trimmed_beta: float = 0.1, krum_f: int = 1,
              q_ffl: float = 2.0, fairfed_beta: float = 1.0,
              state: Dict = None,
              g_target: torch.Tensor = None, g_task: torch.Tensor = None,
              g_fair: torch.Tensor = None, fu_alpha: float = 0.1,
              fu_beta_ema: float = 0.9, fu_normalize: str = "target_norm",
              fu_score: str = "dot", fu_warmup: bool = False,
              fu_grad_clip: float = 0.0, fu_warmup_agg: str = "fedavg",
              bfwa_persist_dual: bool = True,
              gs_tau: float = 1.0, gs_gamma: float = 0.0,
              g_s: torch.Tensor = None, gs_fair: torch.Tensor = None,
              delta_root: torch.Tensor = None, qffl_L: float = None,
              fu_ema_rectify_first: bool = False, fu_magnitude: str = "clip",
              ) -> Tuple[torch.Tensor, Dict]:
    """Aggregate client pseudo-gradients under ``method``.

    ``bfwa_persist_dual`` (default True, matching ``cfg.bfwa_persist_dual``)
    carries the BFWA dual multiplier across communication rounds through
    ``state`` -- the same pattern as ``state['fedgraphfair_lambda']`` and
    ``state['fu_phi_ema']``. ``bfwa`` and ``robust_bfwa`` keep separate keys so
    the two rules never share a multiplier. False restores the pre-fix
    behaviour (the dual restarts from 0 every round, so ``tau`` never binds);
    it exists to keep that regression reproducible, not for reporting.
    """
    K = len(updates)
    stack = torch.stack([u.flatten() for u in updates])
    # Every scalar summary below is built from python floats, so it must be
    # placed on the same device as the updates or GPU runs die in the first
    # weighted sum.
    dev = stack.device
    _t = lambda vals: torch.tensor(vals, dtype=torch.float32, device=dev)
    n = _t([m.get("n", 1) for m in meta])
    perf = _t([m.get("perf", 0.5) for m in meta])
    dpd = _t([m.get("dpd", 0.0) for m in meta])
    loss = _t([m.get("loss", 1.0 - m.get("perf", 0.5)) for m in meta])
    info: Dict = {"method": method}

    if method == "fedavg":
        w = n / n.sum()
        agg = (w[:, None] * stack).sum(0)
        info["weights"] = w.tolist()

    elif method == "fairfed":
        # FairFed (Ezzeldin et al., AAAI 2023): start from data-size weights and
        # down-weight clients whose local fairness gap exceeds the mean gap.
        mean_gap = dpd.mean()
        w = n / n.sum()
        w = w - fairfed_beta * (dpd - mean_gap)
        w = torch.clamp(w, min=0.0)
        w = w / w.sum() if w.sum() > 0 else n / n.sum()
        agg = (w[:, None] * stack).sum(0)
        info["weights"] = w.tolist()

    elif method == "qffl":
        # q-FedAvg (Li et al., ICLR 2020): up-weight high-loss clients for
        # client-level (performance) fairness.
        w = (loss.clamp(min=1e-6)) ** q_ffl
        w = w / w.sum()
        agg = (w[:, None] * stack).sum(0)
        info["weights"] = w.tolist()

    elif method == "qffl_eq":
        # q-FedAvg (Li et al., ICLR 2020) according to upstream litian96/fair_flearn
        # flearn/trainers/qffedavg.py (commit 6112217e59670d6a2f3f9f257147b2c011e40560)
        L = float(qffl_L) if qffl_L is not None else 100.0
        q = float(q_ffl)
        loss_pre_vals = _t([m.get("loss_pre", m.get("loss", 1.0)) for m in meta])
        F_k = loss_pre_vals + 1e-10
        f_pow_q = F_k.pow(q)
        f_pow_qm1 = F_k.pow(q - 1.0)
        norm_l_delta_sq = (L * stack).pow(2).sum(dim=1)
        h_k = q * f_pow_qm1 * norm_l_delta_sq + L * f_pow_q
        h_sum = h_k.sum()
        if h_sum > 0:
            deltas_prime = (f_pow_q * L)[:, None] * stack
            agg = deltas_prime.sum(0) / h_sum
            c_k = (f_pow_q * L) / h_sum
        else:
            agg = torch.zeros_like(stack[0])
            c_k = torch.zeros_like(F_k)
        c_sum = c_k.sum()
        w = c_k / c_sum if c_sum > 0 else n / n.sum()
        info["weights"] = w.tolist()
        info["qffl_scale"] = float(c_sum.item())

    elif method == "f2gnn":
        # F2GNN (Meng et al.): softmax aggregation combining a model-fairness
        # weight (lower DPD -> higher weight) and a data-balance weight (group
        # balance per client), temperature-scaled.
        gbal = _t([1.0 - abs(2.0 * m.get("group1_rate", 0.5) - 1.0) for m in meta])
        gamma_f = torch.softmax(-dpd / 0.1, dim=0)
        gamma_e = torch.softmax(gbal / 0.1, dim=0)
        w = torch.softmax((0.5 * gamma_e + gamma_f) / 0.1, dim=0)
        agg = (w[:, None] * stack).sum(0)
        info["weights"] = w.tolist()

    elif method == "bfwa":
        mu0 = _read_mu(state, "bfwa_mu", bfwa_persist_dual)
        fw_info: Dict = {}
        w = bfwa_weights(perf, dpd, tau, fw_iters, dual_step,
                         mu_init=mu0, info=fw_info)
        _write_mu(state, "bfwa_mu", fw_info["bfwa_mu"], bfwa_persist_dual)
        agg = (w[:, None] * stack).sum(0)
        info["weights"] = w.tolist()
        info.update(fw_info)
        info["tau"] = tau

    elif method == "median":
        agg = stack.median(dim=0).values

    elif method == "trimmed_mean":
        # trim at least krum_f (assumed Byzantine count) from each side
        k = max(int(trimmed_beta * K), krum_f)
        k = min(k, (K - 1) // 2)
        srt = stack.sort(dim=0).values
        agg = srt[k:K - k].mean(0) if K - 2 * k > 0 else srt.mean(0)
        info["trimmed"] = k

    elif method == "krum":
        scores = krum_scores(updates, krum_f)
        sel = int(torch.argmin(scores))
        agg = stack[sel]
        w_full = torch.zeros(K, device=stack.device)
        w_full[sel] = 1.0
        info["selected"] = sel
        info["weights"] = w_full.tolist()

    elif method == "multikrum":
        scores = krum_scores(updates, krum_f)
        m = max(1, K - krum_f)
        sel = torch.argsort(scores)[:m].tolist()
        agg = stack[sel].mean(0)
        w_full = torch.zeros(K, device=stack.device)
        w_full[sel] = 1.0 / len(sel)
        info["selected"] = sel
        info["weights"] = w_full.tolist()

    elif method == "fairgfl":
        # FairGFL (Khan-family overlap-aware reweighting): weight ~
        # 1/(1+overlap_ratio). We proxy the paper's node/edge overlap ratio
        # (a multi-graph-federated quantity we don't have) with each client's
        # normalised deviation from the mean sample count -- a client whose
        # local data looks atypical is analogous to one with an atypical
        # overlap profile. See docs/BASELINES_AND_SOURCES.md.
        het = (n - n.mean()).abs() / (n.mean() + 1e-8)
        w = 1.0 / (1.0 + het)
        w = w / w.sum()
        agg = (w[:, None] * stack).sum(0)
        info["weights"] = w.tolist()

    elif method == "fedgraphfair":
        # FedGraph-Fair (Khan, Information Sciences 2026): the minimax/DRO
        # core -- a simplex-projected dual weight lambda dual-ascended toward
        # clients whose loss exceeds an adaptive cap kappa_max, persisted
        # across rounds via `state`. The paper's personalised-model/graph-
        # mixing layer is not reproduced (single global model here); see
        # docs/BASELINES_AND_SOURCES.md.
        lam = state.get("fedgraphfair_lambda") if state is not None else None
        if lam is None or lam.numel() != K:
            lam = torch.full((K,), 1.0 / K, device=dev)
        lam = lam.to(dev)
        kappa_max = float(loss.mean())
        lam = lam + dual_step * (loss - kappa_max)
        lam = torch.clamp(lam, min=0.0)
        lam = lam / lam.sum() if lam.sum() > 0 else torch.full((K,), 1.0 / K, device=dev)
        if state is not None:
            state["fedgraphfair_lambda"] = lam
        agg = (lam[:, None] * stack).sum(0)
        info["weights"] = lam.tolist()
        info["kappa_max"] = kappa_max

    elif method == "popets_fairfed":
        # PoPETs'25 (Bendoukha et al.): FairFed's weighting made FHE-friendly
        # by replacing exp(-beta|F_i-F_g|) with a degree-2 polynomial (a sign/
        # abs-free surrogate a threshold-CKKS scheme can evaluate). We
        # reproduce only this statistical weighting core; the paper's actual
        # contribution -- a multi-key homomorphic-encryption aggregation
        # protocol -- is systems/crypto infrastructure with no effect on the
        # cleartext numeric result and is not reimplemented (see
        # docs/BASELINES_AND_SOURCES.md).
        F_g = float((n * dpd).sum() / n.sum())
        poly = -fairfed_beta * (dpd - F_g) ** 2 + 1.0
        poly = torch.clamp(poly, min=0.0)
        w_hat = (n / n.sum()) * poly
        w = w_hat / w_hat.sum() if w_hat.sum() > 0 else n / n.sum()
        agg = (w[:, None] * stack).sum(0)
        info["weights"] = w.tolist()
        info["F_g"] = F_g

    elif method == "robust_bfwa":
        # Stage 1: screen out the krum_f updates farthest from the geometric
        # median direction (Byzantine screening).
        med = stack.median(dim=0).values
        dist = ((stack - med) ** 2).sum(1)
        keep = torch.argsort(dist)[: max(1, K - krum_f)].tolist()
        # Stage 2: fairness-constrained Frank-Wolfe among survivors. The dual
        # is keyed separately from plain BFWA: the two rules solve different
        # subproblems (all clients vs survivors) and must not share a
        # multiplier.
        mu0 = _read_mu(state, "robust_bfwa_mu", bfwa_persist_dual)
        fw_info: Dict = {}
        w_sub = bfwa_weights(perf[keep], dpd[keep], tau, fw_iters, dual_step,
                             mu_init=mu0, info=fw_info)
        _write_mu(state, "robust_bfwa_mu", fw_info["bfwa_mu"], bfwa_persist_dual)
        agg = (w_sub[:, None] * stack[keep]).sum(0)
        w_full = torch.zeros(K, device=stack.device)
        w_full[keep] = w_sub
        info["kept"] = keep
        info["survivor_weights"] = w_sub.tolist()
        info["weights"] = w_full.tolist()
        info.update(fw_info)
        info["tau"] = tau

    elif method == "cgsv":
        # CGSV (Xu et al., NeurIPS 2021): cosine-gradient Shapley value. No
        # server validation set -- the reference is the mean client gradient.
        mean_g = stack.mean(0)
        denom = mean_g.norm() + 1e-12
        cos = torch.stack([torch.dot(u, mean_g) / (u.norm() * denom + 1e-12) for u in stack])
        w = torch.relu(cos)
        w = w / w.sum() if float(w.sum()) > 0 else n / n.sum()
        agg = (w[:, None] * stack).sum(0)
        info["weights"] = w.tolist()

    elif method in ("fltrust", "fltrust_ema"):
        # FLTrust (Cao et al., NDSS 2021): the server trains a root update g0 on
        # its own clean dataset, scores each client by ReLU(cosine) against it,
        # and rescales every client update to ||g0|| before the weighted mean.
        #
        # This is the closest structural relative of fu_shapley -- both derive
        # weights from alignment with a server-owned reference and both are
        # therefore metadata-independent -- so it is the baseline that isolates
        # what our bi-objective target actually buys. The one substantive
        # difference is the reference itself: FLTrust's g0 is the TASK gradient
        # only, so a client whose update improves fairness at some task cost
        # gets a *lower* trust score, whereas g_target = g_task + alpha*g_fair
        # In baseline FLTrust, fu_alpha=0.0 so ref is g_task (task gradient only).
        # In the 2x2 ablation grid (P5-X1 / 04_1 §4.6), arm A2 uses FLTrust with
        # fu_alpha > 0, which incorporates g_target = g_task + alpha * g_fair.
        if fu_alpha > 0 and g_target is not None:
            ref = g_target
        else:
            ref = g_task if g_task is not None else g_target

        if ref is None:
            w = _zero_null_players(n / n.sum(), stack)
            agg = (w[:, None] * stack).sum(0)
            info["weights"] = w.tolist(); info["fltrust_fallback"] = "no_root_gradient"
        else:
            ref = ref.to(stack.device).flatten()
            ref_norm = ref.norm() + 1e-12
            cos = torch.stack([torch.dot(u, ref) / (u.norm() * ref_norm + 1e-12) for u in stack])
            ts = torch.relu(cos)
            # FLTrust normalises each client update onto the root's magnitude,
            # which is what bounds a scaling adversary: inflating ||g_k|| cannot
            # buy influence because the norm is divided straight back out.
            scaled = torch.stack([u * (ref_norm / (u.norm() + 1e-12)) for u in stack])
            if method == "fltrust_ema" or (fu_beta_ema > 0 and state is not None and state.get("use_fltrust_ema", False)):
                ts_ema = state.get("fltrust_ts_ema") if state is not None else None
                if ts_ema is None:
                    ts_ema = ts.clone()
                else:
                    ts_ema = fu_beta_ema * ts_ema + (1.0 - fu_beta_ema) * ts
                if state is not None:
                    state["fltrust_ts_ema"] = ts_ema
                ts_eff = ts_ema
            else:
                ts_eff = ts

            tot = float(ts_eff.sum())
            if tot > 0:
                w = ts_eff / ts_eff.sum()
                agg = (w[:, None] * scaled).sum(0)
            else:
                w = torch.zeros_like(ts)
                agg = torch.zeros_like(stack[0])
            info["weights"] = w.tolist()
            info["fltrust_cos"] = cos.tolist()

    elif method == "fltrust_root":
        # FLTrust (Cao et al., NDSS 2021) faithful root-update formulation:
        # Reference is the actual server root update delta_root (task-only, ignoring fu_alpha per ADR-21).
        ref = delta_root if delta_root is not None else g_task
        if ref is None:
            w = torch.zeros(K, device=dev)
            agg = torch.zeros_like(stack[0])
            info["weights"] = w.tolist()
            info["fltrust_fallback"] = "no_root_update"
        else:
            ref = ref.to(stack.device).flatten()
            ref_norm = ref.norm() + 1e-12
            cos = torch.stack([torch.dot(u, ref) / (u.norm() * ref_norm + 1e-12) for u in stack])
            ts = torch.relu(cos)
            scaled = torch.stack([u * (ref_norm / (u.norm() + 1e-12)) for u in stack])
            tot = float(ts.sum())
            if tot > 0:
                w = ts / ts.sum()
                agg = (w[:, None] * scaled).sum(0)
            else:
                w = torch.zeros_like(ts)
                agg = torch.zeros_like(stack[0])
                info["fltrust_fallback"] = "all_zero_trust"
            info["weights"] = w.tolist()
            info["fltrust_cos"] = cos.tolist()

    elif method == "flame":
        # FLAME (Nguyen et al., USENIX Security 2022):
        # 1. Pairwise Cosine Distance matrix between all client updates.
        # 2. Agglomerative Clustering (average linkage) to identify the majority benign cluster.
        # 3. Dynamic Norm Clipping using the median L2 norm of the benign cluster.
        # 4. Uniform averaging over the clipped survivors in the majority cluster.
        if K <= 2:
            w = torch.full((K,), 1.0 / K, device=dev)
            agg = (w[:, None] * stack).sum(0)
            info["weights"] = w.tolist()
            info["selected"] = list(range(K))
        else:
            # Pairwise cosine distance
            norms = stack.norm(dim=1, keepdim=True) + 1e-12
            normed_stack = stack / norms
            cos_sim = torch.mm(normed_stack, normed_stack.t()).clamp(-1.0, 1.0)
            cos_dist = (1.0 - cos_sim).cpu().numpy()
            np.fill_diagonal(cos_dist, 0.0)

            # Hierarchical clustering with average linkage
            from scipy.cluster.hierarchy import linkage, fcluster
            Z = linkage(cos_dist[np.triu_indices(K, k=1)], method="average")
            # Cluster thresholding: dynamically pick threshold or start at 0.5
            cluster_labels = fcluster(Z, t=0.5, criterion="distance")

            # Identify majority cluster
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            max_cluster_label = unique_labels[np.argmax(counts)]
            selected = [i for i, lbl in enumerate(cluster_labels) if lbl == max_cluster_label]

            # Security fallback: if largest cluster is smaller than majority floor, keep all
            if len(selected) < max(1, K // 2 + 1):
                selected = list(range(K))

            # Dynamic median norm clipping over selected benign cluster
            selected_t = torch.tensor(selected, dtype=torch.long, device=dev)
            selected_norms = stack[selected_t].norm(dim=1)
            med_norm = float(selected_norms.median())

            # Clip updates
            clipped_updates = []
            for i in range(K):
                if i in selected:
                    u_norm = float(stack[i].norm() + 1e-12)
                    scale = min(1.0, med_norm / u_norm)
                    clipped_updates.append(stack[i] * scale)
                else:
                    clipped_updates.append(torch.zeros_like(stack[i]))

            clipped_stack = torch.stack(clipped_updates)
            w = torch.zeros(K, device=dev)
            w[selected_t] = 1.0 / len(selected)
            agg = (w[:, None] * clipped_stack).sum(0)

            info["weights"] = w.tolist()
            info["selected"] = selected
            info["flame_med_norm"] = med_norm

    elif method in ("fu_shapley", "robust_fu_shapley"):
        # FairShare-GNN FU-Shapley: score each client against the server target
        # gradient, EMA-smooth, ReLU-gate onto the simplex. See
        # src/trust/incentive.py and implementation_plan_and_ac_review.md PART B.
        from ..trust.incentive import compute_fu_weights, decompose
        if g_target is None:
            # no server validation nodes this round -> fall back to FedAvg
            # (over the non-null clients: see _zero_null_players)
            w = _zero_null_players(n / n.sum(), stack)
            agg = (w[:, None] * stack).sum(0)
            info["weights"] = w.tolist(); info["fu_fallback"] = "no_target"
        else:
            grads = [u for u in stack]
            phi_ema = state.get("fu_phi_ema") if state is not None else None
            fu_info: Dict = {}
            clip_val = None if fu_magnitude == "rescale" else fu_grad_clip
            w, phi_raw, phi_ema_new = compute_fu_weights(
                grads, g_target, phi_ema=phi_ema, beta_ema=fu_beta_ema,
                normalize=fu_normalize, score=fu_score,
                grad_clip=clip_val, rectify_first=fu_ema_rectify_first, info=fu_info)
            if state is not None:
                state["fu_phi_ema"] = phi_ema_new            # thread EMA across rounds
            # SPEC 4.0(d): the status comes from compute_fu_weights itself now.
            # It must not be re-derived from phi_ema_new here: the D2 guard holds
            # the previous (finite) EMA when a score is non-finite, so a NaN
            # round is no longer visible downstream of the guard.
            info["phi_nan_frac"] = fu_info["phi_nan_frac"]
            info["n_clipped"] = fu_info["n_clipped"]
            info["g_norm_median"] = fu_info["g_norm_median"]
            info["g_norm_max"] = fu_info["g_norm_max"]
            info["phi_norm"] = fu_info["phi_norm"]
            info["n_null"] = fu_info["n_null"]
            if fu_info["fu_status"] != "ok":
                info["fu_fallback"] = fu_info["fu_status"]
            robust_median_fallback = False
            keep = None
            if method == "robust_fu_shapley":
                # F4: median-screen the krum_f farthest updates, then re-gate.
                med = stack.median(dim=0).values
                dist = ((stack - med) ** 2).sum(1)
                keep = torch.argsort(dist)[: max(1, K - krum_f)]
                mask = torch.zeros(K, device=dev); mask[keep] = 1.0
                w = w * mask
                if float(w.sum()) > 0:
                    w = w / w.sum()
                else:
                    # DEGENERATE: the screen kept a survivor set, but the FU gate
                    # scored every survivor <= 0. The old fallback here was
                    # `w = n / n.sum()` -- sample-size weights over ALL K clients
                    # -- which hands weight straight back to the very updates the
                    # distance screen had just flagged as Byzantine, so the
                    # robustness of the rule evaporated exactly in the round it
                    # was needed. Fall back to the coordinate-wise median of the
                    # SURVIVORS instead: it is the standard robust estimator, it
                    # needs no contribution score (which is what just failed),
                    # and a screened client contributes nothing to it.
                    # (Uniform-over-survivors was the alternative; the median is
                    # strictly more robust to an outlier that slipped past the
                    # screen and costs nothing, since no weight vector is needed
                    # downstream.)
                    robust_median_fallback = True
                    info["fu_fallback"] = "robust_screen_degenerate_median"
                info["kept"] = keep.tolist()
            warmup_median = False
            if fu_warmup:
                # Warm-up: keep feeding the EMA (done above) but do not let phi
                # set the weights yet (F9).
                #
                # F25: with fu_warmup_agg="fedavg" this window is an attack
                # surface -- the attacker holds ~1/K for the whole window, and
                # under sign_flip a -10x gradient kills the model before the
                # gate ever engages. "median" needs no score history, so it
                # covers the same window without one. Which value to ship is
                # decided by the D11 warm-up ablation, not asserted here.
                if fu_warmup_agg == "median":
                    warmup_median = True
                else:
                    w = _zero_null_players(n / n.sum(), stack)
                # During warm-up the scores are deliberately not trusted, so an
                # all-zero gate is expected rather than degenerate: the robust
                # median fallback (and its flag) do not apply.
                if robust_median_fallback and not warmup_median:
                    robust_median_fallback = False
                    info.pop("fu_fallback", None)
                info["fu_warmup"] = True
                info["fu_warmup_agg"] = fu_warmup_agg

            # ROUND 4 / Gói 4.1 & P5-K10.2 -- bound the PAYLOAD, not just the weight.
            # "rescale": no clipping anywhere. The payload is Δ̃_k = Δ_k · ‖g_target‖ / ‖Δ_k‖,
            # as FLTrust rescales to ‖ref‖. All four C23 cells use fu_score="cosine"
            # (scale-invariant, so the magnitude mode changes only the payload).
            # "clip": the default behaviour (fu_info.get("grads_scored")).
            grads_scored = fu_info.get("grads_scored", None)
            if fu_magnitude == "rescale":
                g_target_flat = g_target.to(dev).flatten()
                target_norm = g_target_flat.norm() + 1e-12
                payload = torch.stack([u * (target_norm / (u.norm() + 1e-12)) for u in stack])
            else:
                payload = stack if grads_scored is None else torch.stack(
                    [p.reshape(-1) for p in grads_scored]).to(stack.dtype).to(dev)
            assert payload.shape == stack.shape, (payload.shape, stack.shape)
            info["effective_updates"] = [p.cpu() for p in payload]

            if warmup_median:
                # Coordinate-wise median is not expressible as client weights,
                # so aggregate directly and report no weight vector.
                agg = stack.median(dim=0).values
                info["weights"] = None
            elif robust_median_fallback:
                # Median of the SURVIVORS only -- screened clients stay out.
                agg = stack[keep].median(dim=0).values
                info["weights"] = None
            else:
                w = _zero_null_players(w, stack)
                agg = (w[:, None] * payload).sum(0)
                info["weights"] = w.tolist()
                info["payload_clipped"] = bool(payload is not stack)
            info["phi_raw"] = phi_raw.tolist()
            info["phi_ema"] = phi_ema_new.tolist()
            if g_task is not None and g_fair is not None:
                # The scores were computed on the CLIPPED gradients (SPEC
                # 4.0(b)); decomposing the raw stack instead broke the
                # explainability identity phi_util + phi_fair == phi_raw in
                # every round where the clip bound.
                scored = grads_scored if grads_scored is not None else grads
                phi_util, phi_fair = decompose(scored, g_task, g_fair, fu_alpha,
                                               score=fu_score)
                info["phi_util"] = phi_util.tolist()
                info["phi_fair"] = phi_fair.tolist()

    elif method == "guardfed_style":
        # GuardFed-style trust score aggregation (Li et al., arXiv 2511.09294 adapted to graphs; ADR-27, K7)
        # 1. Dev_k = ReLU(cos(Delta_k, g_s))
        # 2. Fair_k = EO gap of theta - Delta_k on holdout (passed in gs_fair)
        # 3. Trust_k = Dev_k * exp(-gs_tau * Fair_k)
        # 4. S = {k : Trust_k > gs_gamma}
        # 5. Update = uniform average of Delta_k over S. If S is empty, agg = 0 (theta unchanged).
        # 6. w_adv = 1/|S| if adversary in S, else 0.
        if g_s is None:
            w = _zero_null_players(n / n.sum(), stack)
            agg = (w[:, None] * stack).sum(0)
            info["weights"] = w.tolist()
            info["selected"] = list(range(K))
        else:
            g_s_flat = g_s.to(dev).flatten()
            gs_norm = g_s_flat.norm() + 1e-12
            cos = torch.stack([torch.dot(u, g_s_flat) / (u.norm() * gs_norm + 1e-12) for u in stack])
            dev_scores = torch.relu(cos)

            fair_scores = gs_fair.to(dev).flatten() if gs_fair is not None else torch.zeros(K, device=dev)
            trust_scores = dev_scores * torch.exp(-gs_tau * fair_scores)

            selected = [k for k in range(K) if float(trust_scores[k]) > gs_gamma]
            w = torch.zeros(K, device=dev)
            if len(selected) > 0:
                for k in selected:
                    w[k] = 1.0 / len(selected)
                agg = (w[:, None] * stack).sum(0)
            else:
                agg = torch.zeros_like(stack[0])

            info["weights"] = w.tolist()
            info["selected"] = selected
            info["trust_scores"] = trust_scores.tolist()
            info["dev_scores"] = dev_scores.tolist()
            info["fair_scores"] = fair_scores.tolist()

    else:
        raise ValueError(f"Unknown aggregator '{method}'")

    return agg.view_as(updates[0]), info


ROBUST_METHODS = {"krum", "multikrum", "median", "trimmed_mean", "robust_bfwa",
                  "robust_fu_shapley", "flame", "fltrust", "fltrust_ema"}
FAIR_METHODS = {"bfwa", "fairfed", "qffl", "f2gnn", "fairgfl", "fedgraphfair",
                "popets_fairfed", "fu_shapley", "robust_fu_shapley"}
ALL_METHODS = {"fedavg", "cgsv", "guardfed_style"} | FAIR_METHODS | ROBUST_METHODS

