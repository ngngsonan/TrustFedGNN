"""Experiment configuration and reproducibility utilities.

All experiments in this repository are driven by an :class:`ExperimentConfig`
dataclass so that every reported number can be traced back to an exact,
serialisable configuration. This is a deliberate design choice for a research
artifact: the manuscript's tables and figures are regenerated from logged runs,
not hand-authored.
"""
from __future__ import annotations

import dataclasses
import json
import os
import random
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch


@dataclass
class ExperimentConfig:
    """Full specification of a federated experiment.

    Grouped by concern: data / partition / model / federated optimisation /
    fairness / privacy / robustness / trust. Defaults reproduce the primary
    configuration reported in the manuscript.
    """

    # ---- experiment identity ----
    exp_name: str = "default"
    seed: int = 42
    # Compute device. Default is CPU; set FEDFAIR_DEVICE=cuda to move every run
    # onto the GPU without touching call sites (no auto-detection on purpose --
    # the device is part of the environment manifest and must be declared).
    device: str = field(default_factory=lambda: os.environ.get("FEDFAIR_DEVICE", "cpu"))
    out_dir: str = "results"

    # ---- data ----
    dataset: str = "german"          # german | credit | bail | pokec_z | pokec_n | synthetic
    data_root: str = "data"
    label_number: int = 1000         # max labelled train nodes per NIFTY convention (small datasets ignore)

    # ---- federated partition ----
    num_clients: int = 5
    partition: str = "dirichlet"     # dirichlet | uniform | community
    dirichlet_alpha: float = 0.5     # smaller alpha => more non-IID
    partition_by: str = "label"      # label | sensitive  (what the Dirichlet skews)

    # ---- model ----
    model: str = "trustfedgnn"       # trustfedgnn | gcn | gat | fairgnn | ...
    hidden_channels: int = 64
    num_layers: int = 2
    heads: int = 4
    dropout: float = 0.3

    # ---- mini-batch neighbor sampling (for graphs too large for full-batch) ----
    sampling: bool = False
    batch_size: int = 2048
    num_neighbors: tuple = (15, 10)  # neighbours sampled per GNN layer
    eval_max_nodes: int = 0          # cap test nodes used per evaluation (0 = all);
                                     # a random subsample keeps evaluation tractable on
                                     # million-node graphs whose test split is enormous

    # ---- federated optimisation ----
    rounds: int = 100
    local_epochs: int = 3
    local_lr: float = 0.01
    weight_decay: float = 1e-5

    # ---- aggregation ----
    aggregator: str = "bfwa"         # fedavg | bfwa | krum | multikrum | median | trimmed_mean | robust_bfwa
                                     #   | fu_shapley | robust_fu_shapley | cgsv | guardfed_style   (FairShare-GNN + baseline)
    trimmed_beta: float = 0.1        # fraction trimmed each side for trimmed_mean
    krum_f: int = 1                  # assumed number of Byzantine clients for Krum/Multi-Krum
    gs_tau: float = 1.0              # GuardFed-style exponential penalty on equal opportunity gap (ADR-27, K7)
    gs_gamma: float = 0.0            # GuardFed-style threshold for survivor inclusion (ADR-27, K7)

    # ---- FU-Shapley incentive aggregation (FairShare-GNN) ----
    #   Server scores each client g_k against a target gradient built on the pooled
    #   validation set: g_target = g_task + fu_alpha * g_fair  (see src/trust/incentive.py).
    #   Sign is '+': g_fair is the *ascent* gradient of the fairness surrogate, so a
    #   client that descends fairness loss has g_k aligned with g_fair -> positive credit.
    fu_alpha: float = 0.1            # fairness trade-off weight in the target gradient (F1)
    fu_ema_beta: float = 0.9         # EMA decay for temporal Shapley smoothing (F6/F9)
    fu_warmup_rounds: int = 0        # rounds before FU-SV gating; EMA is still fed (F9).
                                     #   Was 5. The D11 warm-up ablation (R7) shows
                                     #   nan_round_frac under sign_flip rises monotonically with
                                     #   this window under "fedavg" -- 0 / .067 / .200 / .467 for
                                     #   0 / 3 / 5 / 10 rounds -- because the attacker holds ~1/K
                                     #   until the gate engages. 0 and warmup_agg="median" both
                                     #   drive it to zero; AUC cannot separate them (seed range
                                     #   0.113 at n=3). 0 wins on parsimony: it removes a knob
                                     #   rather than adding a second defence whose credit would
                                     #   not attribute to FU-Shapley. Warm-up is absent from the
                                     #   maths, so dropping it touches no claim.
    fu_warmup_agg: str = "fedavg"    # {fedavg, median} -- how to aggregate DURING warm-up.
                                     #   F25: "fedavg" leaves the attacker ~1/K for the whole
                                     #   window, long enough for sign_flip to kill the model
                                     #   before the gate engages. "median" needs no score
                                     #   history so it covers the window. Ship value is decided
                                     #   by the D11 warm-up ablation, not asserted here.
    fu_normalize: str = "target_norm"  # per-round phi scaling: none | target_norm | zscore.
                                     #   target_norm (default) divides by ||g_target|| -> fixes cross-
                                     #   round scale drift while PRESERVING sign (F6). zscore recenters
                                     #   (drops ~half the clients, degenerate for small K) -> ablation only.
    fu_score: str = "dot"            # contribution score: dot | cosine (cosine = CGSV-style, F4 ablation)
    fu_fair_surrogate: str = "sq"    # server fairness target surrogate: sq=(mu0-mu1)^2 | abs (F5)
    fu_grad_clip: float = 10.0       # SPEC 4.0(b): cap ||g_k|| before scoring. A defence-side bound on
                                     #   single-client influence -- it does NOT weaken the adversary
                                     #   (attack_intensity is untouched). 0 disables.
    fu_val_source: str = "pooled"    # pooled | server_holdout. 'pooled' builds g_target from every
                                     #   client's val nodes INCLUDING Byzantine ones, which assumes away
                                     #   the very threat being studied (A1/F10). 'server_holdout' carves
                                     #   fu_holdout_size nodes out before partitioning so no client owns
                                     #   them. See src/trust/incentive.py.
    fu_holdout_size: int = 200       # nodes REQUESTED for the server when fu_val_source='server_holdout'.
                                     #   C16: this is a request, not a guarantee. carve_server_holdout caps
                                     #   it at half the validation split, so on german (250 val nodes) any
                                     #   request above 125 is clipped -- the default 200 already is. The
                                     #   number actually scored against is holdout.granted_size /
                                     #   holdout.num_nodes, and a clipped request now warns. Read that,
                                     #   never this field, when reporting a D7/D8 holdout-size ablation.
    fu_holdout_mode: str = "stratified"  # stratified | connected (ADR-27, K8)
    fu_ema_rectify_first: bool = False   # rectify score before EMA in compute_fu_weights (P5-K10.1)
    fu_magnitude: str = "clip"           # clip | rescale (P5-K10.2)
    log_mechanism: bool = False          # log detailed mechanism metrics in trainer.history (P5-K10.3)
    pokec_label_protocol: str = "legacy_negative"  # legacy_negative | known_only (P5-K10.4)
    skip_client_meta: bool = False   # F7: actually skip client meta() forward pass (only then may we
                                     #   claim client compute savings). Default keeps meta() for logging.

    # ---- fairness ----
    fairness_weight: float = 1.0     # lambda: weight of fairness loss
    fairness_budget: float = 0.05    # tau: max allowed global DPD in BFWA
    beta_init: float = 0.5           # FSER coefficient init
    fser_mode: str = "sub"           # sub (cross_penalize canonical) | add (cross_boost) | same_penalize
    freeze_beta: bool = False        # hold FSER beta at beta_init (no gradient). With
                                     #   beta_init=0.0 this is the ONLY faithful "w/o FSER"
                                     #   ablation: FSERLayer at beta=0 is exactly GAT attention,
                                     #   so the backbone (BN + residual + skip-concat) is held
                                     #   fixed. The `ours-nofser` arm swaps model="gat" instead,
                                     #   which removes FSER *and* the whole scaffold at once and
                                     #   therefore attributes architecture gains to FSER.
    fw_iterations: int = 20
    dual_step_size: float = 0.1
    bfwa_persist_dual: bool = True   # carry the BFWA dual multiplier mu across rounds.
                                     #   False reproduces the pre-fix behaviour (mu reset to 0
                                     #   every call), under which tau never binds -- kept only
                                     #   so the regression is reproducible, never for reporting.

    # ---- privacy (DP) ----
    dp_enabled: bool = True
    dp_epsilon: float = 8.0
    dp_delta: float = 1e-5
    dp_clip: float = 1.0
    dp_mode: str = "auto"            # auto | none | ftgd | gradient
    #   ftgd     : privatise fairness statistics (ours)
    #   gradient : standard full-gradient DP-SGD (DP-FedAvg baseline)
    dp_statistic_s_blind: bool = True  # compute the RELEASED group means from an s-blind forward
                                     #   pass (sensitive_attr=None). Required for the sensitivity
                                     #   bound to hold: FSER makes yhat a function of s, so
                                     #   flipping one node's group perturbs predictions across its
                                     #   whole L-hop neighbourhood and Delta is NOT O(1/n_min).
                                     #   With this on, the released statistic is computed by a
                                     #   forward pass that never reads s, restoring yhat _|_ s for
                                     #   the released quantity while FSER still runs in the task
                                     #   pathway. False = pre-fix behaviour; the reported epsilon
                                     #   is then unsupported for the FSER backbone.
    report_privatised_dpd: bool = True  # meta() reports the DP-noised disparity actually produced
                                     #   by _ftgd_step instead of a raw val-split DPD. Off, the
                                     #   server consumes un-noised fairness statistics and the
                                     #   "privacy-aware feedback loop" is not implemented.
    report_group_rate: bool = False  # transmit the client's raw sensitive-attribute marginal
                                     #   (group1_rate) to the server. Default OFF: it is an
                                     #   un-noised, unaccounted s-derived statistic released every
                                     #   round. Only FairFed-style aggregators need it.
    report_loss_pre: bool = False    # compute and report pre-training task BCE on global model for qffl_eq

    # ---- generic fairness (for baselines that add a local DP penalty) ----
    local_fairness: bool = False     # add soft-DPD penalty for non-trustfedgnn models
    q_ffl: float = 2.0               # q for q-FedAvg client-fairness aggregation
    fairfed_beta: float = 1.0        # FairFed fairness-gap step
    postproc_fair: bool = False      # FDP-Fair: post-hoc group-offset calibration for DP

    # ---- 2025/2026 baseline-specific knobs ----
    puffle_target_dpd: float = 0.05  # PUFFLE: target disparity T the lambda controller tracks
    puffle_momentum: float = 0.9     # PUFFLE: controller momentum
    puffle_rho: float = 0.5          # PUFFLE: controller step size
    fedfact_post: bool = False       # FedFACT: two-level (global lambda + local mu_k) post-hoc offset
    fedfact_local_scale: float = 1.0 # FedFACT: weight on the per-client local offset vs the global one

    # ---- EquFL (Yu et al., arXiv:2601.05352, 2026): server-side fairness calibration ----
    server_calib: bool = False           # add a server-side fairness-gradient correction each round
    server_calib_start_frac: float = 0.5 # start calibrating after this fraction of rounds (EquFL default)
    server_calib_gamma: float = 1.0      # weight on the calibration gradient g0

    # ---- FTGD gradient surgery ----
    ftgd_projection: str = "always"  # always | conflict.
                                     #   always   : project the task gradient orthogonal to the
                                     #              fairness gradient unconditionally (as published).
                                     #   conflict : project only when <g_task, g_fair> < 0, i.e.
                                     #              the PCGrad rule (Yu et al., NeurIPS'20). When
                                     #              the objectives agree, 'always' deletes a
                                     #              cooperative component PCGrad keeps -- strictly
                                     #              slower on the task for no fairness gain. This
                                     #              is the ablation that isolates that choice.
    ftgd_min_fair_norm: float = 1e-8 # skip the projection when ||g_fair|| falls below this and
                                     #   record it, instead of letting a numerical epsilon in the
                                     #   denominator silently scale the projection to nothing.

    # ---- robustness (attack simulation) ----
    attack: str = "none"             # none | label_flip | gaussian | scaling | fairness_poison
                                     #   | ipm | alie
    num_byzantine: int = 0
    attack_intensity: float = 10.0
    # Which aggregator the adversary tailors its false report to. None =
    # the legacy three-field lie. See attacks.BEST_RESPONSE_LIE.
    meta_lie: Optional[str] = None   # scale for gaussian / scaling / sign_flip / fairness_poison.
                                     #   NOT used by ipm or alie -- those have their own calibrated
                                     #   parameters below, because a large intensity destroys the
                                     #   stealth that defines both attacks.
    ipm_epsilon: float = 0.5         # IPM (Xie et al.): g_byz = -eps * mean(benign). The attack
                                     #   REQUIRES eps < 1 so the update stays inside the benign
                                     #   radius; at the old attack_intensity=10.0 it degenerated
                                     #   into plain scaling and lost its point.
    alie_z: Optional[float] = None   # ALIE (Baruch et al.): g_byz = mean - z*std. None (default)
                                     #   solves z from the benign quantile given (K, f), which is
                                     #   the actual attack; a hardcoded value ignores K and f and
                                     #   typically sits far outside the benign range.

    # ---- logging ----
    wandb: bool = False
    wandb_project: str = "fedfairgnn"
    wandb_entity: Optional[str] = None
    save_dir: str = "checkpoints"

    # ---- uncertainty / trust ----
    mc_dropout_samples: int = 20     # forward passes for MC-dropout uncertainty
    eval_uncertainty: bool = False

    # ---- bookkeeping ----
    notes: str = ""

    def run_id(self) -> str:
        return f"{self.exp_name}__{self.model}__{self.dataset}__{self.aggregator}__seed{self.seed}"

    def __setattr__(self, name, value):
        """Reject assignment to anything that is not a declared field.

        A plain dataclass silently accepts ``cfg.some_typo = True``, and a runner
        that toggles an ablation with a flag no code reads produces two arms that
        are byte-identical -- an experiment that measures nothing while appearing
        to measure a defence. That happened here: a Byzantine sweep switched its
        "two-tier defence" on and off via ``cfg.fu_cosine_filter`` and
        ``cfg.fu_multikrum``, neither of which exists on this class nor is read
        anywhere in src/, so both arms were the same configuration and the
        hypothesis was scored on floating-point noise.

        Failing loudly here makes that class of null experiment impossible. Use a
        real field (add it to this dataclass and read it) rather than stashing
        state on the config.
        """
        if name not in _FIELD_NAMES:
            raise AttributeError(
                f"ExperimentConfig has no field {name!r}; assigning it would "
                "create a knob that no code reads. Declare it as a dataclass "
                f"field and consume it, or fix the name. Known fields: "
                f"{sorted(_FIELD_NAMES)}")
        object.__setattr__(self, name, value)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def canonical(cls, **overrides) -> "ExperimentConfig":
        """The ONLY configuration of the 'Ours' arm (TrustFedGNN).
        
        Every table in the paper and every ablation is initialised from here.
        Ablation = canonical(...) with an explicit override that matches the arm name.
        """
        base = dict(
            dataset="german",
            num_clients=5,
            rounds=20,
            dirichlet_alpha=0.3,
            model="trustfedgnn",
            aggregator="fu_shapley",
            fu_alpha=0.1,
            fu_ema_beta=0.9,
            fairness_weight=1.0,
            beta_init=0.5,
            fser_mode="sub",
            dp_enabled=True,
            dp_mode="ftgd",
            fu_val_source="server_holdout",
        )
        base.update(overrides)
        return cls(**base)


# Declared-field allowlist for ExperimentConfig.__setattr__. Computed after the
# dataclass is built (the decorator needs to run first) and before any instance
# exists, since the generated __init__ assigns through __setattr__.
_FIELD_NAMES = frozenset(f.name for f in dataclasses.fields(ExperimentConfig))


def set_seed(seed: int) -> None:
    """Seed all RNGs for deterministic behaviour.

    Note: differential-privacy noise is *intentionally* stochastic; results are
    reported as mean +/- std over multiple seeds (see experiments/).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(False)  # some PyG scatter ops lack deterministic CUDA kernels
