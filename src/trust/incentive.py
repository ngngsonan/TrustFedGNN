"""FU-Shapley incentive aggregation for FairShare-GNN.

The server evaluates each client's contribution *without trusting any self-
reported metadata* by aligning the client pseudo-gradient ``g_k = theta_global -
theta_local_k`` against a *target gradient* computed independently on the pooled
validation set:

    g_target = g_task + alpha * g_fair                            (FU-SV target)
    phi_k    = <g_k, g_target>                                    (contribution)
    w_k      = ReLU(EMA(phi_k)) / sum_j ReLU(EMA(phi_j))          (simplex weight)

Design decisions locked after the area-chair review (see
``../../implementation_plan_and_ac_review.md`` PART A):

* **F1 — sign is PLUS.** ``g_fair = +grad L_fair`` is the *ascent* gradient of
  the fairness surrogate. A client that *descends* fairness loss (reduces the
  demographic-parity gap) produces ``g_k`` aligned with ``g_fair`` -> positive
  fairness credit. The proposal's original minus sign would have penalised
  exactly the fairness-improving clients.
* **F2 — coordinate layout follows ``state_dict()``**, zero-filling entries with
  no gradient (BatchNorm buffers). This makes ``g_target`` line up coordinate-
  for-coordinate with ``flatten_state(state_dict())`` used to build ``g_k`` in
  the trainer. Building the target from ``model.parameters()`` alone would give
  a shorter, mis-ordered vector.
* **F3 — the task target uses the client's own loss** (``_weighted_bce``) on the
  ``val_mask``, so ``phi_util`` measures alignment with the *same* objective the
  clients optimise.
* **F5 — the server fairness surrogate defaults to ``(mu0 - mu1)**2``** (smooth),
  whose gradient vanishes gracefully as fairness is achieved. The client keeps
  the non-smooth ``|mu0 - mu1|`` unchanged.
* **F6 — per-round scaling defaults to ``target_norm``** (divide by
  ``||g_target||``): it removes the cross-round scale drift (as ``g_task -> 0``
  near convergence) while *preserving the sign* of ``phi``, so the ReLU gate
  keeps meaning "actively harmful" rather than "below average". ``zscore`` is
  offered for ablation only -- it recentres and therefore always gates roughly
  half the clients (and is degenerate for small K).

This module is intentionally NOT re-exported from ``trust/__init__`` to keep the
package import graph acyclic (it imports from ``federated.client``).
"""
from __future__ import annotations

import copy
import warnings
from typing import List, Optional, Tuple

import torch

from ..federated.client import _weighted_bce, flatten_state


# --------------------------------------------------------------------------- #
# Gradient plumbing
# --------------------------------------------------------------------------- #
def _flat_from_grad_map(model, grad_map: dict) -> torch.Tensor:
    """Flatten grads given as a ``{param_name: grad_or_None}`` map into
    ``state_dict()`` order, zero-filling keys without a gradient (buffers) and
    casting to float32 so the result lines up with ``flatten_state`` (F2)."""
    parts: List[torch.Tensor] = []
    for k, v in model.state_dict().items():
        g = grad_map.get(k)
        part = (g.detach().flatten() if g is not None
                else torch.zeros(v.numel(), device=v.device))
        parts.append(part.to(torch.float32))
    return torch.cat(parts)


def _flat_grad_like_state(model) -> torch.Tensor:
    """Flatten ``model``'s current ``.grad`` in ``state_dict()`` order (F2)."""
    return _flat_from_grad_map(model, {n: p.grad for n, p in model.named_parameters()})


def _missing_group(s: torch.Tensor, where: str) -> bool:
    """True (with a RuntimeWarning) if either sensitive group is absent from the
    scoring set.

    The demographic-parity surrogate ``(mu0 - mu1)**2`` needs both group means.
    The previous code substituted ``mu_g = 0`` for an empty group, which turns
    the objective into ``mu_present**2`` -- not a fairness loss at all, and
    silently so. Callers must drop the fairness term for that round instead;
    ``alpha * g_fair`` is then exactly zero rather than corrupt, and this
    warning records which group went missing.
    """
    has0 = bool((s == 0).any())
    has1 = bool((s == 1).any())
    if has0 and has1:
        return False
    missing = "s=0" if not has0 else "s=1"
    warnings.warn(
        f"FU-Shapley target gradient: sensitive group {missing} is absent from "
        f"the scoring set ({where}, n={int(s.numel())} nodes). The "
        f"demographic-parity surrogate is undefined there, so the fairness term "
        f"is DROPPED for this round (g_fair = 0, g_target = g_task) rather than "
        f"substituting mu_{missing} = 0, which would optimise a different "
        f"objective. Fairness credit (phi_fair) is 0 for every client this "
        f"round.",
        RuntimeWarning, stacklevel=3)
    return True


def get_server_target_gradients(model, data_val, alpha: float, *,
                                fair_surrogate: str = "sq"
                                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (g_target, g_task, g_fair) on the pooled validation set.

    Args:
        model: current global model (weights already loaded).
        data_val: PyG ``Data`` with ``val_mask``, ``sensitive_attr``.
        alpha: fairness trade-off weight in ``g_target = g_task + alpha*g_fair``.
        fair_surrogate: ``"sq"`` -> ``(mu0-mu1)**2`` (default, smooth, F5);
                        ``"abs"`` -> ``|mu0-mu1|`` (matches the client metric).

    Returns:
        g_target, g_task, g_fair -- flat float32 tensors laid out like
        ``flatten_state(model.state_dict())`` (F2).
    """
    model.eval()   # deterministic forward: no dropout noise in the reference
    x, ei, s = data_val.x, data_val.edge_index, data_val.sensitive_attr
    m = data_val.val_mask                                     # F3: val, not train
    y = data_val.y

    # --- task target: same weighted BCE the clients minimise (F3) ---
    model.zero_grad(set_to_none=True)
    pred = model(x, ei, s)[m]
    loss_task = _weighted_bce(pred, y[m].float())
    loss_task.backward()
    g_task = _flat_grad_like_state(model)

    # --- fairness target: smooth soft-DPD surrogate (F5) ---
    model.zero_grad(set_to_none=True)
    pred = model(x, ei, s)
    s_m = s[m]
    pm = pred[m]
    if _missing_group(s_m, "server holdout (val_mask)"):
        # D-degeneracy: with one group absent the demographic-parity surrogate
        # is undefined, and the previous `mu_g = 0` substitution silently
        # optimised `mu_present**2` -- a different objective that pushes every
        # prediction of the present group to 0. Drop the term instead.
        g_fair = torch.zeros_like(g_task)
    else:
        mu0 = pm[s_m == 0].mean()
        mu1 = pm[s_m == 1].mean()
        loss_fair = (mu0 - mu1) ** 2 if fair_surrogate == "sq" else torch.abs(mu0 - mu1)
        loss_fair.backward()
        g_fair = _flat_grad_like_state(model)

    g_target = g_task + alpha * g_fair                        # F1: PLUS sign
    model.zero_grad(set_to_none=True)
    return g_target, g_task, g_fair


def get_server_root_update(model, data_holdout, cfg, t: int) -> torch.Tensor:
    """Compute server root update on holdout data for FLTrust (Cao et al., NDSS 2021).

    Runs local_epochs full-batch steps of task-only BCE training on the server holdout
    starting from theta_{t-1}, using the client's effective learning rate and weight decay.
    Returns flat(theta_{t-1}) - flat(theta_after) matching client pseudo-gradient convention.
    Wrapped in torch.random.fork_rng(devices=[]) with manual_seed(cfg.seed * 1000 + t).
    """
    cuda_devs = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devs):
        torch.manual_seed(int(cfg.seed) * 1000 + int(t))
        model_copy = copy.deepcopy(model)
        model_copy.train()
        lr = float(cfg.local_lr)
        weight_decay = float(cfg.weight_decay)
        opt = torch.optim.AdamW(model_copy.parameters(), lr=lr, weight_decay=weight_decay)

        x, ei, s = data_holdout.x, data_holdout.edge_index, data_holdout.sensitive_attr
        m = data_holdout.val_mask if hasattr(data_holdout, "val_mask") and data_holdout.val_mask is not None else torch.ones(data_holdout.num_nodes, dtype=torch.bool, device=x.device)
        y = data_holdout.y

        flat_before = flatten_state(model.state_dict()).cpu()

        for _ in range(cfg.local_epochs):
            opt.zero_grad()
            pred = model_copy(x, ei, s)[m]
            loss = _weighted_bce(pred, y[m].float())
            loss.backward()
            opt.step()
            if hasattr(model_copy, "clamp_beta"):
                model_copy.clamp_beta()

        flat_after = flatten_state(model_copy.state_dict()).cpu()
        return flat_before - flat_after


def get_server_target_gradients_pooled(model, clients_data, device, alpha: float,
                                       *, fair_surrogate: str = "sq"):
    """Target gradients over the *pooled* client validation nodes.

    In this cross-silo setup each client owns a distinct subgraph, so there is no
    single server graph: we forward each client's ``val_mask`` nodes through the
    (current global) model, concatenate the predictions, and take one backward
    per objective -- mirroring the existing EquFL server calibration
    (``trainer._server_calibration_grad``). A single forward feeds both the task
    and fairness objectives via ``retain_graph`` so the two targets are exactly
    consistent.

    Returns ``(g_target, g_task, g_fair)`` in ``state_dict`` layout (F2), or
    ``None`` if no client exposes validation nodes.
    """
    model.eval()
    preds, ys, ss = [], [], []
    for d in clients_data:
        d = d.to(device)
        m = d.val_mask
        if int(m.sum()) == 0:
            continue
        preds.append(model(d.x, d.edge_index, d.sensitive_attr)[m])
        ys.append(d.y[m]); ss.append(d.sensitive_attr[m])
    if not preds:
        return None
    pred = torch.cat(preds); y = torch.cat(ys); s = torch.cat(ss)

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]

    loss_task = _weighted_bce(pred, y.float())                           # F3
    gt = torch.autograd.grad(loss_task, params, retain_graph=True, allow_unused=True)
    g_task = _flat_from_grad_map(model, {n: g for (n, _), g in zip(named, gt)})

    if _missing_group(s, "pooled client validation nodes"):
        # See get_server_target_gradients: no group contrast -> no fairness
        # signal. Zero is the honest value; `mu_missing = 0` was not.
        g_fair = torch.zeros_like(g_task)
    else:
        mu0 = pred[s == 0].mean()
        mu1 = pred[s == 1].mean()
        loss_fair = (mu0 - mu1) ** 2 if fair_surrogate == "sq" else torch.abs(mu0 - mu1)  # F5
        gf = torch.autograd.grad(loss_fair, params, allow_unused=True)
        g_fair = _flat_from_grad_map(model, {n: g for (n, _), g in zip(named, gf)})
    g_target = g_task + alpha * g_fair                                   # F1: PLUS
    return g_target, g_task, g_fair


def fairness_gradient_ratio(g_task, g_fair, alpha: float) -> float:
    """Computes ||alpha * g_fair|| / (||g_task|| + 1e-12) as a diagnostic ratio."""
    def _norm(g):
        if isinstance(g, dict):
            return torch.sqrt(sum((v ** 2).sum() for v in g.values())).item()
        elif isinstance(g, torch.Tensor):
            return g.norm().item()
        return 0.0
    norm_task = _norm(g_task)
    norm_fair = _norm(g_fair)
    return float(alpha * norm_fair / (norm_task + 1e-12))


# --------------------------------------------------------------------------- #
# Contribution scoring
# --------------------------------------------------------------------------- #
def _score(g_k: torch.Tensor, ref: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "cosine":
        return torch.dot(g_k, ref) / (g_k.norm() * ref.norm() + 1e-12)
    return torch.dot(g_k, ref)


def _normalize(phi_raw: torch.Tensor, g_target: torch.Tensor,
               mode: str) -> torch.Tensor:
    """Per-round rescaling of the raw scores (F6). ``target_norm`` preserves
    sign; ``zscore`` recentres (ablation only)."""
    if mode == "target_norm":
        return phi_raw / (g_target.norm() + 1e-8)
    if mode == "zscore":
        return (phi_raw - phi_raw.mean()) / (phi_raw.std() + 1e-8)
    return phi_raw


def _clip_grads(grads: List[torch.Tensor], max_norm: float) -> Tuple[List[torch.Tensor], int]:
    """Rescale any ``g_k`` whose L2 norm exceeds ``max_norm`` (SPEC 4.0(b)).

    This is a *defence-side* bound on how much influence a single client can
    exert on the score, not a weakening of the adversary: ``attack_intensity``
    is untouched. Non-finite updates cannot be clipped meaningfully and are
    passed through unchanged for the finiteness guard below to catch."""
    out, n_clipped = [], 0
    for g in grads:
        nrm = g.norm()
        if torch.isfinite(nrm) and float(nrm) > max_norm:
            g = g * (max_norm / (float(nrm) + 1e-12))
            n_clipped += 1
        out.append(g)
    return out, n_clipped


def compute_fu_weights(client_grads: List[torch.Tensor], g_target: torch.Tensor,
                       *, phi_ema: Optional[torch.Tensor] = None,
                       beta_ema: float = 0.9, normalize: str = "target_norm",
                       score: str = "dot", grad_clip: Optional[float] = None,
                       rectify_first: bool = False,
                       info: Optional[dict] = None
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FU-Shapley weights: score -> per-round normalise -> EMA -> ReLU-simplex.

    Args:
        client_grads: list of K flat pseudo-gradients ``g_k``.
        g_target: target gradient from :func:`get_server_target_gradients`.
        phi_ema: EMA state from the previous round (``None`` on the first).
        beta_ema: EMA decay; ``beta_ema=0`` disables smoothing.
        normalize: ``none`` | ``target_norm`` | ``zscore`` (F6).
        score: ``dot`` | ``cosine`` (F4).
        grad_clip: if set, cap ``||g_k||`` at this value before scoring
                   (SPEC 4.0(b)); ``None`` disables.
        rectify_first: if True, feed relu(phi_n) into EMA (P5-K10.1).
        info: optional dict filled in-place with diagnostics -- ``phi_nan_frac``,
              ``n_clipped``, ``n_null``, ``fu_status`` in
              ``{ok, degenerate_nan, degenerate_nonpos}`` (SPEC 4.0(d)), and
              ``grads_scored`` (the post-clip gradients the scores were computed
              on, to be passed to :func:`decompose`).

    Returns:
        weights: K-vector on the simplex (sums to 1). Clients with EMA-smoothed
                 score <= 0 receive exactly 0; if *all* are <= 0 the rule falls
                 back to uniform weights over the clients whose update is not
                 identically zero (null-player axiom: a client that submitted
                 nothing gets 0 in every branch, fallbacks included).
        phi_raw: unnormalised per-round scores (for logging / decomposition).
        phi_ema_new: updated EMA state to thread into the next round -- always
                 finite (see below), so it is safe to thread indefinitely.

    Finiteness contract (proposal 1.3.1 D1-D3). ``phi`` is only defined for
    finite ``g_k`` and ``g_target``; under ``sign_flip`` the exploding gradient
    makes it NaN. The EMA recursion has NaN as an *absorbing* state -- once
    ``phi_ema`` is NaN it stays NaN forever, even after ``phi_raw`` recovers --
    and because ``nan > 0`` is False the rule then silently degrades to FedAvg,
    handing the attacker a full ``1/K`` share. So: coordinates whose score is
    non-finite this round **hold their previous EMA value** instead of being
    updated (D2), and on the very first round they are set to 0, i.e. gated out
    rather than given uniform credit -- an undefined contribution earns none.
    """
    K = len(client_grads)
    # Scale of ||g_k|| BEFORE clipping. Logging norms separates "no outlier" from
    # "threshold in the wrong units". Note: empirical validation in Tier C (T3)
    # refuted the prior speculation that `fu_grad_clip=10.0` was inert on clean data;
    # on German benign data, c=10.0 actively bounds 59.8% of client updates (T3 negative branch).
    _norms = sorted(float(g.norm()) for g in client_grads
                    if torch.isfinite(g.norm()))
    g_norm_median = _norms[len(_norms) // 2] if _norms else float("nan")
    g_norm_max = _norms[-1] if _norms else float("nan")

    # Null players are identified BEFORE clipping (rescaling never changes
    # whether a vector is exactly zero, but the intent is "what the client
    # actually submitted").
    null = torch.tensor([float(g.abs().max()) == 0.0 for g in client_grads],
                        device=g_target.device)

    if grad_clip is not None and grad_clip > 0:
        client_grads, n_clipped = _clip_grads(client_grads, grad_clip)
    else:
        n_clipped = 0

    phi_raw = torch.stack([_score(g, g_target, score) for g in client_grads])
    phi_n = _normalize(phi_raw, g_target, normalize)
    phi_in = torch.relu(phi_n) if rectify_first else phi_n

    # --- D1/D2: never let a non-finite score enter (or poison) the EMA -------
    finite = torch.isfinite(phi_in)
    if phi_ema is None:
        phi_ema_new = torch.where(finite, phi_in, torch.zeros_like(phi_in))
    else:
        updated = beta_ema * phi_ema + (1.0 - beta_ema) * phi_in
        phi_ema_new = torch.where(finite, updated, phi_ema)
    # Belt-and-braces: a NaN threaded in from an older checkpoint must not leak.
    phi_ema_new = torch.nan_to_num(phi_ema_new, nan=0.0, posinf=0.0, neginf=0.0)

    # --- null-player axiom ---------------------------------------------------
    # A client whose update is exactly zero contributed nothing: its score is
    # <0, g_target> = 0 under every score mode, so the ReLU gate already gives
    # it weight 0. The *fallback* did not: when every score was non-positive the
    # rule handed uniform 1/K to all K clients, null players included, which is
    # precisely the axiom violation (a null player must receive exactly 0 in
    # every branch, not merely in the branch that happens to look at phi). The
    # mask below enforces that for both branches, and also covers the case where
    # a client goes null after banking positive EMA history in earlier rounds.
    active = ~null
    clip = torch.relu(phi_ema_new) * active.to(phi_ema_new.dtype)
    total = clip.sum()
    n_active = int(active.sum())
    if total > 0:
        weights = clip / total
        status = "ok"
    else:
        if n_active > 0:
            # Uniform over the NON-degenerate clients only.
            weights = active.to(phi_raw.dtype) / n_active
        else:
            # Every client is null: the aggregate is the zero vector whatever
            # the weights are, so uniform keeps the simplex contract.
            weights = torch.full((K,), 1.0 / K, device=phi_raw.device)
        # D3: "measurement broke" and "everyone contributed negatively" produce
        # the same weights but are different states -- label them apart.
        status = "degenerate_nan" if not bool(finite.all()) else "degenerate_nonpos"

    if info is not None:
        info["phi_nan_frac"] = round(float((~finite).sum()) / K, 4)
        info["n_clipped"] = n_clipped
        info["fu_status"] = status
        info["g_norm_median"] = g_norm_median
        info["g_norm_max"] = g_norm_max
        # phi AFTER normalisation, BEFORE the EMA. Without it the FS-1b contract
        # cannot be checked from the logs: `phi_raw` is the unnormalised score,
        # while the EMA recursion runs on `phi_n`, and the ||g_target|| scaling
        # changes every round -- so `phi_ema` cannot be re-derived from
        # `phi_raw` alone.
        info["phi_norm"] = [round(float(v), 8) for v in phi_n]
        info["n_null"] = K - n_active
        # The gradients the scores were ACTUALLY computed on (post-clip). The
        # caller must feed these -- not the raw stack -- to `decompose`, or
        # phi_util + phi_fair != phi_raw in every round where the clip binds.
        # Tensors, not a log field: `aggregate` copies only scalars into `info`.
        info["grads_scored"] = client_grads
    return weights, phi_raw, phi_ema_new


def decompose(client_grads: List[torch.Tensor], g_task: torch.Tensor,
              g_fair: torch.Tensor, alpha: float, *, score: str = "dot"
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explainable split ``phi_k = phi_util + phi_fair`` (F1: PLUS sign).

        phi_util = <g_k, g_task>          (utility contribution)
        phi_fair = alpha * <g_k, g_fair>  (fairness contribution; >0 = reduces
                                           disparity, since g_fair is ascent grad)
    """
    phi_util = torch.stack([_score(g, g_task, score) for g in client_grads])
    phi_fair = torch.stack([alpha * _score(g, g_fair, score) for g in client_grads])
    return phi_util, phi_fair
