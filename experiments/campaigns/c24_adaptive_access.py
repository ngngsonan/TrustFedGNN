"""C24 Campaign Runner: Adaptive Access Sweeps (K15, round 3).

Investigates defence resilience under constrained adversary access schedules:
  - S1: Static / delayed poison: honest for rounds 0..9, fairness_poison for t >= 10.
  - S2: Dynamic / periodic poison: honest for rounds 0..9, fairness_poison when
        t >= 10 and ((t - 10) // 5) is even; honest otherwise.
  - S3: Inference attack from t >= 1: crafts adversarial update deceive FU/FLTrust
        using only public theta_history, own honest update, own data, and model template.
  - Benign: num_byzantine = 0 (clean baseline for every method).

Methods evaluated:
  1. fedavg: canonical FedAvg
  2. fltrust_root: faithful root-update FLTrust (Cao et al. NDSS'21)
  3. fu_shapley: canonical bi-objective FU-Shapley (cosine score, clip magnitude)
  4. guardfed_style: parameter-matched baseline using C22 selected configuration
     (German: gs_tau=1.0, gs_gamma=0.03; Bail: gs_tau=1.0, gs_gamma=0.0)

Datasets: German, Bail (K = 10 clients, R = 40 rounds, Dirichlet alpha = 0.3)
Byzantine fraction: f/K = 0.2 (2 adversaries), optionally f/K = 0.4
Seeds: 42..51 (10 seeds)
Mechanisms logged: log_mechanism=True (I_adv, w_adv_round, cos_agg_g_task, cos_agg_g_fair)

Artifact: results/campaigns/C24_adaptive_access.json
"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Force CPU execution and isolate threads for multiprocessing
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from src.config import ExperimentConfig, set_seed
from src.federated.client import load_flat_state
from src.federated.trainer import FederatedTrainer
from src.utils.provenance import build_manifest

DATASETS = ["german", "bail"]
METHODS = ["fedavg", "fltrust_root", "fu_shapley", "guardfed_style"]
SCHEDULES = ["S1", "S2", "S3"]
SEEDS_CONFIRM = list(range(42, 52))
PILOT_SEED = 2000
NUM_CLIENTS = 10
ROUNDS = 40
LOCAL_EPOCHS = 3
DIRICHLET_ALPHA = 0.3
ATTACK_INTENSITY = 10.0
OPT_LR = 0.01
OPT_STEPS = 50
LAMBDA_FAIR = 1.0


def get_method_kwargs(method: str, dataset: str) -> Dict[str, Any]:
    """Get method-specific configuration dictionary."""
    if method == "fedavg":
        return dict(aggregator="fedavg")
    elif method == "fltrust_root":
        return dict(
            aggregator="fltrust_root",
            fu_alpha=0.0,
            fu_val_source="server_holdout",
            fu_holdout_size=200,
        )
    elif method == "fu_shapley":
        return dict(
            aggregator="fu_shapley",
            fu_alpha=0.1,
            fu_score="cosine",
            fu_magnitude="clip",
            fu_ema_rectify_first=False,
            fu_val_source="server_holdout",
            fu_holdout_size=200,
        )
    elif method == "guardfed_style":
        # Configurations selected in C22 Part A for German and Bail
        if dataset == "german":
            gs_tau = 1.0
            gs_gamma = 0.03
        else:
            gs_tau = 1.0
            gs_gamma = 0.0
        return dict(
            aggregator="guardfed_style",
            gs_tau=gs_tau,
            gs_gamma=gs_gamma,
            fu_val_source="server_holdout",
            fu_holdout_size=200,
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def craft_s3_adversary_update(
    theta_history: List[torch.Tensor],
    own_update: torch.Tensor,
    own_data: Any,
    model_template: torch.nn.Module,
    lambda_fair: float = LAMBDA_FAIR,
    opt_steps: int = OPT_STEPS,
    opt_lr: float = OPT_LR,
) -> torch.Tensor:
    """Craft S3 adversarial update under strict information access constraints (K15 / B.2).

    The adversary receives ONLY:
      1. theta_history: list of broadcast global model states up to round t
      2. own_update: honest update of adversary from its own local training
      3. own_data: local private dataset of the adversary
      4. model_template: public architecture template of the model

    Objective replaces C10 alignment adversary with three substitutions:
      (1) g_t := r_hat_t = theta_{t-1} - theta_t (applied update from previous round)
      (2) g_f := adversary's own fairness gradient on train nodes at theta_t (L_fair = (mu0 - mu1)^2)
      (3) initial value and reference norm := own honest update.
    """
    if len(theta_history) < 2:
        raise ValueError("S3 attack requires at least 2 consecutive global models in theta_history (t >= 1)")

    # 1. g_t := r_hat_t = theta_{t-1} - theta_t
    r_hat_t = (theta_history[-2] - theta_history[-1]).detach().float().cpu()
    r_norm = float(r_hat_t.norm().item())
    if r_norm < 1e-12:
        g_t_norm = torch.zeros_like(r_hat_t)
    else:
        g_t_norm = r_hat_t / r_norm

    # 2. g_f := adversary's own fairness gradient on train nodes at theta_t
    device = torch.device("cpu")
    model = copy.deepcopy(model_template).to(device)
    theta_t = theta_history[-1].detach().to(device)
    load_flat_state(model, theta_t)
    model.train()

    x = own_data.x.to(device)
    ei = own_data.edge_index.to(device)
    s = own_data.sensitive_attr.to(device)
    m = own_data.train_mask.to(device)

    pred = model(x, ei, s)[m]
    s_m = s[m]

    m0 = (s_m == 0)
    m1 = (s_m == 1)

    if m0.sum() > 0 and m1.sum() > 0:
        mu0 = pred[m0].mean()
        mu1 = pred[m1].mean()
        l_fair = (mu0 - mu1) ** 2
        model.zero_grad()
        l_fair.backward()
        # Flatten gradients in state_dict order zero-filling buffers to match flatten_state
        parts = []
        grad_map = {n: p.grad for n, p in model.named_parameters()}
        for k, v in model.state_dict().items():
            g = grad_map.get(k)
            part = g.detach().flatten() if g is not None else torch.zeros(v.numel(), device=v.device)
            parts.append(part.to(torch.float32))
        g_f = torch.cat(parts).detach().float().cpu()
        g_f_norm_val = float(g_f.norm().item())
        if g_f_norm_val < 1e-12:
            g_f_norm = -g_t_norm
        else:
            g_f_norm = g_f / g_f_norm_val
    else:
        g_f_norm = -g_t_norm

    # 3. Initial value and reference norm := own honest update
    own_up = own_update.clone().detach().float().cpu()
    g_adv = own_up.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([g_adv], lr=opt_lr)
    ref_norm = float(own_up.norm().item())

    for _ in range(opt_steps):
        optimizer.zero_grad()
        g_adv_norm = F.normalize(g_adv.unsqueeze(0), dim=1).squeeze(0)
        align_target = (g_adv_norm * g_t_norm).sum()
        harm_fair = (g_adv_norm * g_f_norm).sum()

        loss = -align_target + lambda_fair * harm_fair
        norm_loss = torch.relu(g_adv.norm() - ref_norm * 5.0)
        total_loss = loss + 0.1 * norm_loss
        total_loss.backward()
        optimizer.step()

    return g_adv.detach()


class AdaptiveAccessTrainer(FederatedTrainer):
    """FederatedTrainer with adaptive access attacker schedules (K15 / B.2).

    Schedules (rounds 0-indexed):
      - S1: poison when t >= 10.
      - S2: poison when t >= 10 and ((t - 10) // 5) % 2 == 0.
      - S3: inference attack from t >= 1 using craft_s3_adversary_update.
      - Benign: num_byzantine = 0.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        schedule: str = "S3",
        lambda_fair: float = LAMBDA_FAIR,
        opt_steps: int = OPT_STEPS,
        opt_lr: float = OPT_LR,
    ):
        super().__init__(cfg)
        self.schedule = schedule
        self.lambda_fair = lambda_fair
        self.opt_steps = opt_steps
        self.opt_lr = opt_lr
        if self.cfg.num_byzantine > 0 and not self.byzantine_ids:
            self.byzantine_ids = list(range(min(self.cfg.num_byzantine, len(self.clients))))
            for b_id in self.byzantine_ids:
                self.clients[b_id].byzantine = True
        self.theta_history: List[torch.Tensor] = [self.global_flat.clone().detach().cpu()]

    def _is_poison_round(self, t: int) -> bool:
        """Evaluate if round t is an active adversarial poisoning round."""
        if not self.byzantine_ids:
            return False
        if self.schedule == "S1":
            return t >= 10
        elif self.schedule == "S2":
            return (t >= 10) and (((t - 10) // 5) % 2 == 0)
        elif self.schedule == "S3":
            return t >= 1
        elif self.schedule == "benign":
            return False
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def _round(self, t: int) -> dict:
        is_poison = self._is_poison_round(t)

        updates: List[Optional[torch.Tensor]] = [None] * len(self.clients)
        metas: List[Optional[dict]] = [None] * len(self.clients)

        # 1. Broadcast global model
        for c in self.clients:
            c.set_flat(self.global_flat)

        t0_clients = time.perf_counter()

        # 2. Honest clients train
        for i, c in enumerate(self.clients):
            if i not in self.byzantine_ids:
                c.train()
                g_k = self.global_flat - c.get_flat()
                updates[i] = g_k
                metas[i] = c.meta()

        # 3. Byzantine clients train
        if self.byzantine_ids:
            if not is_poison:
                # Benign round: update and report MUST BE IDENTICAL to honest client
                for b_id in self.byzantine_ids:
                    c = self.clients[b_id]
                    prev_byz = c.byzantine
                    c.byzantine = False
                    c.train()
                    c.byzantine = prev_byz
                    g_k = self.global_flat - c.get_flat()
                    updates[b_id] = g_k
                    metas[b_id] = c.meta()
            else:
                # Poison round
                if self.schedule in ("S1", "S2"):
                    # Standard fairness_poison attack (intensity 10)
                    for b_id in self.byzantine_ids:
                        c = self.clients[b_id]
                        c.byzantine = True
                        c.train()
                        g_k = self.global_flat - c.get_flat()
                        updates[b_id] = g_k
                        m = c.meta()
                        m.update({"dpd": 0.0, "eod": 0.0, "perf": 0.99})
                        metas[b_id] = m
                elif self.schedule == "S3":
                    # S3 inference attack: primary adversary computes honest update
                    primary_c = self.clients[self.byzantine_ids[0]]
                    prev_byz = primary_c.byzantine
                    primary_c.byzantine = False
                    primary_c.train()
                    primary_c.byzantine = prev_byz
                    own_update = self.global_flat - primary_c.get_flat()

                    # Craft S3 update using only allowed information
                    g_adv = craft_s3_adversary_update(
                        theta_history=self.theta_history,
                        own_update=own_update,
                        own_data=primary_c.data,
                        model_template=self.ref_model,
                        lambda_fair=self.lambda_fair,
                        opt_steps=self.opt_steps,
                        opt_lr=self.opt_lr,
                    )

                    # All adversaries send identical update and C10 metadata
                    for b_id in self.byzantine_ids:
                        updates[b_id] = g_adv.clone()
                        m = self.clients[b_id].meta()
                        # The report keeps the client's true sample count `n`
                        # (aggregate() reads `n`; the threat model never lets an
                        # adversary misreport it) and claims the same utility as
                        # the S1/S2 lie.
                        metas[b_id] = {
                            "n": m["n"],
                            "perf": 0.99,
                            "dpd": 0.0,
                            "loss": 0.01,
                            "group1_rate": 0.5,
                        }

        t_clients_s = time.perf_counter() - t0_clients

        if self.accountant:
            self.accountant.step(self.cfg.local_epochs)

        # 4. Server references and aggregation
        t0_server = time.perf_counter()
        g_target = g_task = g_fair = None
        fu_warmup = False
        if "fu_shapley" in self.cfg.aggregator or self.cfg.aggregator == "fltrust":
            load_flat_state(self.ref_model, self.global_flat.to(self.device))
            from src.trust.incentive import (get_server_target_gradients,
                                           get_server_target_gradients_pooled)
            if self.server_holdout is not None:
                tg = get_server_target_gradients(
                    self.ref_model, self.server_holdout.to(self.device),
                    self.cfg.fu_alpha, fair_surrogate=self.cfg.fu_fair_surrogate)
            else:
                tg = get_server_target_gradients_pooled(
                    self.ref_model, self.clients_data, self.device,
                    self.cfg.fu_alpha, fair_surrogate=self.cfg.fu_fair_surrogate)
            if tg is not None:
                g_target, g_task, g_fair = (g.cpu() for g in tg)
            fu_warmup = t < self.cfg.fu_warmup_rounds

        delta_root = None
        if self.cfg.aggregator == "fltrust_root":
            from src.trust.incentive import get_server_root_update
            delta_root = get_server_root_update(
                self.ref_model, self.server_holdout, self.cfg, t)

        g_s = gs_fair = None
        if self.cfg.aggregator == "guardfed_style":
            from src.federated.aggregation import compute_guardfed_holdout
            holdout = self.server_holdout.to(self.device) if self.server_holdout is not None else None
            if holdout is not None:
                g_s, gs_fair = compute_guardfed_holdout(
                    self.ref_model, self.global_flat, updates, holdout, self.device
                )

        qffl_L = 1.0 / self.cfg.lr if self.cfg.aggregator == "qffl_eq" else 1.0

        from src.federated.aggregation import aggregate
        g_agg, info = aggregate(
            self.cfg.aggregator, updates, metas,
            tau=self.cfg.fairness_budget, fw_iters=self.cfg.fw_iterations,
            dual_step=self.cfg.dual_step_size, trimmed_beta=self.cfg.trimmed_beta,
            krum_f=max(self.cfg.krum_f, len(self.byzantine_ids)),
            q_ffl=self.cfg.q_ffl, fairfed_beta=self.cfg.fairfed_beta,
            state=self._agg_state,
            g_target=g_target, g_task=g_task, g_fair=g_fair,
            fu_alpha=self.cfg.fu_alpha, fu_beta_ema=self.cfg.fu_ema_beta,
            fu_normalize=self.cfg.fu_normalize, fu_score=self.cfg.fu_score,
            fu_warmup=fu_warmup, fu_grad_clip=self.cfg.fu_grad_clip,
            fu_warmup_agg=self.cfg.fu_warmup_agg,
            bfwa_persist_dual=self.cfg.bfwa_persist_dual,
            gs_tau=self.cfg.gs_tau, gs_gamma=self.cfg.gs_gamma,
            g_s=g_s, gs_fair=gs_fair,
            delta_root=delta_root, qffl_L=qffl_L,
            fu_ema_rectify_first=getattr(self.cfg, "fu_ema_rectify_first", False),
            fu_magnitude=getattr(self.cfg, "fu_magnitude", "clip"))
        t_server_s = time.perf_counter() - t0_server

        if self.cfg.server_calib and (t + 1) >= self.cfg.rounds * self.cfg.server_calib_start_frac:
            g0 = self._server_calibration_grad()
            if g0 is not None:
                g_agg = g_agg + self.cfg.server_calib_gamma * g0

        self.global_flat = self.global_flat - g_agg
        self.theta_history.append(self.global_flat.clone().detach().cpu())

        rec = {"round": t + 1, **{f"g_{k}": v for k, v in self.evaluate_global().items()}}
        rec["agg_weights"] = info.get("weights")
        for key in ("phi_raw", "phi_util", "phi_fair", "phi_ema", "fu_warmup", "kept",
                    "fu_fallback", "phi_nan_frac", "n_clipped", "n_null",
                    "g_norm_median", "g_norm_max", "phi_norm"):
            if key in info:
                rec[key] = info[key]

        # 5. Mechanism logging (with corrected I_adv from c696a4f)
        if getattr(self.cfg, "log_mechanism", False):
            g_task_eval = g_task
            g_fair_eval = g_fair
            if g_task_eval is None or g_fair_eval is None:
                cuda_devs = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
                with torch.random.fork_rng(devices=cuda_devs):
                    load_flat_state(self.ref_model, self.global_flat.to(self.device))
                    from src.trust.incentive import (get_server_target_gradients,
                                                   get_server_target_gradients_pooled)
                    if self.server_holdout is not None:
                        tg_eval = get_server_target_gradients(
                            self.ref_model, self.server_holdout.to(self.device),
                            self.cfg.fu_alpha, fair_surrogate=self.cfg.fu_fair_surrogate)
                    else:
                        tg_eval = get_server_target_gradients_pooled(
                            self.ref_model, self.clients_data, self.device,
                            self.cfg.fu_alpha, fair_surrogate=self.cfg.fu_fair_surrogate)
                    if tg_eval is not None:
                        _, g_task_eval, g_fair_eval = (g.cpu() for g in tg_eval)

            if g_task_eval is not None:
                g_task_f = g_task_eval.flatten()
                norm_prod = g_agg.norm() * g_task_f.norm() + 1e-12
                cos_task = float((torch.dot(g_agg, g_task_f) / norm_prod).item())
            else:
                cos_task = None

            if g_fair_eval is not None:
                g_fair_f = g_fair_eval.flatten()
                norm_prod = g_agg.norm() * g_fair_f.norm() + 1e-12
                cos_fair = float((torch.dot(g_agg, g_fair_f) / norm_prod).item())
            else:
                cos_fair = None

            w_vec = info.get("weights")
            if w_vec is not None and self.byzantine_ids:
                w_adv_round = float(sum(w_vec[i] for i in self.byzantine_ids if i < len(w_vec)))
            elif w_vec is not None:
                w_adv_round = 0.0
            else:
                w_adv_round = None

            if info.get("effective_updates") is not None:
                eff_updates = [p.flatten().cpu() for p in info["effective_updates"]]
            elif self.cfg.aggregator in ("fltrust", "fltrust_ema", "fltrust_root"):
                if self.cfg.aggregator == "fltrust_root":
                    ref_scale = delta_root if delta_root is not None else g_task
                else:
                    if getattr(self.cfg, "fu_alpha", 0.0) > 0 and g_target is not None:
                        ref_scale = g_target
                    else:
                        ref_scale = g_task if g_task is not None else g_target
                if ref_scale is not None:
                    ref_norm = float(ref_scale.flatten().norm().item()) + 1e-12
                    eff_updates = [
                        u.flatten().cpu() * (ref_norm / (float(u.flatten().norm().item()) + 1e-12))
                        for u in updates
                    ]
                else:
                    eff_updates = [u.flatten().cpu() for u in updates]
            else:
                eff_updates = [u.flatten().cpu() for u in updates]

            if w_vec is not None and self.byzantine_ids:
                adv_terms = [
                    float(w_vec[i]) * eff_updates[i].float()
                    for i in self.byzantine_ids
                    if i < len(w_vec) and i < len(eff_updates)
                ]
                if adv_terms:
                    I_adv = float(torch.stack(adv_terms).sum(dim=0).norm().item())
                else:
                    I_adv = 0.0
            elif w_vec is not None:
                I_adv = 0.0
            else:
                I_adv = None

            phi_now = info.get("phi_raw")
            if phi_now is None:
                phi_now = info.get("fltrust_cos")
            if phi_now is None:
                phi_now = info.get("trust_scores")

            phi_ema_val = info.get("phi_ema")
            if phi_ema_val is None and self._agg_state is not None:
                ts_ema = self._agg_state.get("fltrust_ts_ema")
                if ts_ema is not None:
                    phi_ema_val = ts_ema.tolist() if hasattr(ts_ema, "tolist") else ts_ema

            fallback = info.get("fu_fallback") or info.get("fltrust_fallback")

            rec.update({
                "adv_ids": list(self.byzantine_ids),
                "w": w_vec,
                "phi_now": phi_now,
                "phi_ema": phi_ema_val,
                "fallback": fallback,
                "w_adv_round": w_adv_round,
                "I_adv": I_adv,
                "agg_norm": float(g_agg.norm().item()),
                "cos_agg_g_task": cos_task,
                "cos_agg_g_fair": cos_fair,
                "t_server_s": t_server_s,
                "t_clients_s": t_clients_s,
            })

        return rec


def run_one(
    dataset: str,
    method: str,
    schedule: str,
    byzantine_fraction: float,
    seed: int,
    rounds: int = ROUNDS,
    device: str = "cpu",
) -> dict:
    """Execute a single C24 experiment run with isolated worker threads."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    num_byz = int(NUM_CLIENTS * byzantine_fraction)
    method_kwargs = get_method_kwargs(method, dataset)

    attack_kwargs = dict(
        attack="fairness_poison" if num_byz > 0 else "none",
        num_byzantine=num_byz,
        attack_intensity=ATTACK_INTENSITY if num_byz > 0 else 0.0,
    )

    cfg = ExperimentConfig.canonical(
        dataset=dataset,
        seed=seed,
        num_clients=NUM_CLIENTS,
        rounds=rounds,
        local_epochs=LOCAL_EPOCHS,
        dirichlet_alpha=DIRICHLET_ALPHA,
        device=device,
        dp_enabled=False,
        log_mechanism=True,
        **method_kwargs,
        **attack_kwargs,
    )

    t0 = time.perf_counter()
    trainer = AdaptiveAccessTrainer(cfg, schedule=schedule)
    res = trainer.run(verbose=False)
    wall = time.perf_counter() - t0
    fin = res.get("final", res)

    # Calculate mean w_adv and I_adv over rounds
    w_adv_list = []
    i_adv_list = []
    per_round = []

    for rec in getattr(trainer, "history", []):
        w_round = rec.get("w_adv_round")
        if w_round is not None:
            w_adv_list.append(float(w_round))
        i_round = rec.get("I_adv")
        if i_round is not None:
            i_adv_list.append(float(i_round))

        pr = {
            "round": rec.get("round"),
            "w": rec.get("w"),
            "phi_now": rec.get("phi_now"),
            "phi_ema": rec.get("phi_ema"),
            "fallback": rec.get("fallback"),
            "w_adv_round": w_round,
            "I_adv": i_round,
            "agg_norm": rec.get("agg_norm"),
            "cos_agg_g_task": rec.get("cos_agg_g_task"),
            "cos_agg_g_fair": rec.get("cos_agg_g_fair"),
            "t_server_s": rec.get("t_server_s"),
            "t_clients_s": rec.get("t_clients_s"),
        }
        per_round.append(pr)

    mean_w_adv = float(np.mean(w_adv_list)) if w_adv_list else 0.0
    mean_i_adv = float(np.mean(i_adv_list)) if i_adv_list else 0.0

    return {
        "dataset": dataset,
        "method": method,
        "schedule": schedule,
        "byzantine_fraction": byzantine_fraction,
        "seed": seed,
        "auc": float(fin["auc"]),
        "dpd_hard": float(fin["dpd_hard"]),
        "eod": float(fin.get("eod", 0.0)),
        "w_adv": mean_w_adv,
        "I_adv": mean_i_adv,
        "wall_s": float(wall),
        "per_round": per_round,
    }


def _worker_task(args: Tuple) -> dict:
    return run_one(*args)


def check_c24_completeness(results: dict, expected_cells: list, expected_seeds: list[int]):
    """Ensure every expected cell has all required seeds. Raises RuntimeError on any gap."""
    for ds, sch, m, bf in expected_cells:
        cell_key = f"{ds}__{sch}__{m}__bf{bf}"
        cell_data = results.get(ds, {}).get(sch, {}).get(m, {})
        recs = cell_data.get("per_seed", [])
        seeds_present = {r["seed"] for r in recs}
        missing = set(expected_seeds) - seeds_present
        if missing:
            raise RuntimeError(
                f"C24 completeness check FAIL: {cell_key} is missing seeds: {sorted(missing)} "
                f"(expected {len(expected_seeds)} seeds, found {len(seeds_present)})"
            )


def run_pilot(workers: int = 6) -> dict:
    """Run pilot (seed 2000, f/K=0.2, R=40) across all cells to record timing and estimate budget."""
    print(f"=== Starting C24 Pilot Sweep (seed {PILOT_SEED}, f/K=0.2, R={ROUNDS}, workers={workers}) ===")
    t_start = time.perf_counter()

    pilot_cells = []
    # 1. Attacked cells at f/K = 0.2: 2 datasets x 3 schedules x 4 methods = 24 cells
    for ds in DATASETS:
        for sch in SCHEDULES:
            for m in METHODS:
                pilot_cells.append((ds, m, sch, 0.2, PILOT_SEED, ROUNDS, "cpu"))

    # 2. Benign cells (num_byzantine = 0): 2 datasets x 4 methods = 8 cells
    for ds in DATASETS:
        for m in METHODS:
            pilot_cells.append((ds, m, "benign", 0.0, PILOT_SEED, ROUNDS, "cpu"))

    print(f"[*] Total pilot cells: {len(pilot_cells)} runs")

    if workers > 1:
        with mp.Pool(processes=workers) as pool:
            results = pool.map(_worker_task, pilot_cells)
    else:
        results = [_worker_task(c) for c in pilot_cells]

    total_wall = time.perf_counter() - t_start

    timing_table = {}
    for r in results:
        key = f"{r['dataset']}__{r['schedule']}__{r['method']}"
        timing_table[key] = {
            "wall_s": r["wall_s"],
            "auc": r["auc"],
            "dpd_hard": r["dpd_hard"],
            "w_adv": r["w_adv"],
            "I_adv": r["I_adv"],
        }

    # Calculate budget projections
    # Priority order: (1) S3 f/K=0.2 + benign -> (2) S1 -> (3) S2 -> (4) f/K=0.4
    sec_s3_benign = sum(timing_table[f"{ds}__{sch}__{m}"]["wall_s"]
                        for ds in DATASETS for m in METHODS for sch in ("S3", "benign"))
    sec_s1 = sum(timing_table[f"{ds}__S1__{m}"]["wall_s"] for ds in DATASETS for m in METHODS)
    sec_s2 = sum(timing_table[f"{ds}__S2__{m}"]["wall_s"] for ds in DATASETS for m in METHODS)

    est_10seed_s3_benign_hours = (sec_s3_benign * 10) / (workers * 3600.0)
    est_10seed_s1_hours = (sec_s1 * 10) / (workers * 3600.0)
    est_10seed_s2_hours = (sec_s2 * 10) / (workers * 3600.0)

    total_all_3_schedules_hours = est_10seed_s3_benign_hours + est_10seed_s1_hours + est_10seed_s2_hours

    summary = {
        "pilot_seed": PILOT_SEED,
        "rounds": ROUNDS,
        "workers": workers,
        "total_pilot_wall_s": total_wall,
        "timing_table": timing_table,
        "projections_hours": {
            "p1_s3_plus_benign": est_10seed_s3_benign_hours,
            "p2_s1": est_10seed_s1_hours,
            "p3_s2": est_10seed_s2_hours,
            "total_all_3_schedules": total_all_3_schedules_hours,
        },
    }

    print("\n=== PILOT TIMING & BUDGET PROJECTION ===")
    print(f"Pilot completed in {total_wall:.1f}s ({total_wall/60.0:.2f} min)")
    print(f"Est. 10 seeds (workers={workers}):")
    print(f"  P1 (S3 + benign): {est_10seed_s3_benign_hours:.2f} hours")
    print(f"  P2 (S1):          {est_10seed_s1_hours:.2f} hours")
    print(f"  P3 (S2):          {est_10seed_s2_hours:.2f} hours")
    print(f"  Total (S1+S2+S3+benign): {total_all_3_schedules_hours:.2f} hours (Budget limit: 10.0 hours)")

    return summary


def run_campaign(
    out_json: str,
    schedules: Optional[List[str]] = None,
    include_f04: bool = False,
    smoke: bool = False,
    workers: int = 6,
    methods: Optional[List[str]] = None,
    include_benign: bool = True,
    tag: str = "C24",
):
    """Run confirmatory C24 sweep with checkpointing and atomic writing."""
    t_start = time.perf_counter()
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    chk_path = out_json.replace(".json", "_checkpoint.json")

    target_schedules = schedules or (["S3"] if smoke else SCHEDULES)
    seeds = [42] if smoke else SEEDS_CONFIRM
    rounds = 2 if smoke else ROUNDS
    datasets = ["german"] if smoke else DATASETS
    methods = ["fu_shapley"] if smoke else (methods or METHODS)

    print(f"=== Starting C24 Campaign (smoke={smoke}, workers={workers}, rounds={rounds}) ===")
    print(f"Schedules: {target_schedules}, Datasets: {datasets}, Methods: {methods}, Seeds: {len(seeds)}")

    # Define tasks
    expected_cells = []
    tasks = []

    # 1. Attacked runs
    for ds in datasets:
        for sch in target_schedules:
            for m in methods:
                expected_cells.append((ds, sch, m, 0.2))
                for s in seeds:
                    tasks.append((ds, m, sch, 0.2, s, rounds, "cpu"))
                if include_f04 and not smoke:
                    expected_cells.append((ds, sch, m, 0.4))
                    for s in seeds:
                        tasks.append((ds, m, sch, 0.4, s, rounds, "cpu"))

    # 2. Benign runs (clean baseline)
    for ds in (datasets if include_benign else []):
        for m in methods:
            expected_cells.append((ds, "benign", m, 0.0))
            for s in seeds:
                tasks.append((ds, m, "benign", 0.0, s, rounds, "cpu"))

    # Load existing checkpoint if available
    cached_recs = {}
    if os.path.exists(chk_path) and not smoke:
        try:
            with open(chk_path) as f:
                chk_data = json.load(f)
            for r in chk_data.get("raw_runs", []):
                k = (r["dataset"], r["method"], r["schedule"], r["byzantine_fraction"], r["seed"])
                cached_recs[k] = r
            print(f"[*] Resumed from checkpoint: {len(cached_recs)} tasks already completed.")
        except Exception as e:
            print(f"[!] Warning reading checkpoint: {e}")

    tasks_to_run = [t for t in tasks if (t[0], t[1], t[2], t[3], t[4]) not in cached_recs]
    print(f"[*] Tasks to execute: {len(tasks_to_run)} / {len(tasks)}")

    finished_recs = list(cached_recs.values())

    if tasks_to_run:
        if workers > 1:
            with mp.Pool(processes=workers) as pool:
                for res in pool.imap_unordered(_worker_task, tasks_to_run):
                    finished_recs.append(res)
                    # Checkpoint atomic write
                    if not smoke and len(finished_recs) % workers == 0:
                        tmp_chk = chk_path + ".tmp"
                        with open(tmp_chk, "w") as f:
                            json.dump({"raw_runs": finished_recs}, f)
                        os.replace(tmp_chk, chk_path)
        else:
            for t in tasks_to_run:
                res = _worker_task(t)
                finished_recs.append(res)
                if not smoke:
                    tmp_chk = chk_path + ".tmp"
                    with open(tmp_chk, "w") as f:
                        json.dump({"raw_runs": finished_recs}, f)
                    os.replace(tmp_chk, chk_path)

    # Build structured hierarchical results dictionary
    structured: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in finished_recs:
        ds = r["dataset"]
        sch = r["schedule"]
        m = r["method"]
        bf = r["byzantine_fraction"]

        structured.setdefault(ds, {}).setdefault(sch, {}).setdefault(m, {"per_seed": []})
        structured[ds][sch][m]["per_seed"].append(r)

    # Calculate summaries
    for ds in structured:
        for sch in structured[ds]:
            for m in structured[ds][sch]:
                recs = structured[ds][sch][m]["per_seed"]
                # Sort per_seed deterministically by seed
                recs.sort(key=lambda x: x["seed"])
                structured[ds][sch][m]["per_seed"] = recs
                structured[ds][sch][m]["n_seeds"] = len(recs)
                structured[ds][sch][m]["mean_auc"] = float(np.mean([x["auc"] for x in recs]))
                structured[ds][sch][m]["mean_dpd"] = float(np.mean([x["dpd_hard"] for x in recs]))
                structured[ds][sch][m]["mean_eod"] = float(np.mean([x["eod"] for x in recs]))
                structured[ds][sch][m]["mean_w_adv"] = float(np.mean([x["w_adv"] for x in recs]))
                structured[ds][sch][m]["mean_I_adv"] = float(np.mean([x["I_adv"] for x in recs]))

    # Gating completeness check (raises RuntimeError if missing)
    if not smoke:
        check_c24_completeness(structured, expected_cells, seeds)

    total_wall = time.perf_counter() - t_start
    manifest = build_manifest()

    final_payload = {
        "manifest": manifest,
        "meta": {
            "campaign": tag,
            "title": "Adaptive Access Sweeps",
            "smoke": smoke,
            "workers": workers,
            "rounds": rounds,
            "datasets": datasets,
            "methods": methods,
            "schedules": target_schedules,
            "seeds": seeds,
            "wall_clock_s": total_wall,
        },
        "results": structured,
        "raw_runs": finished_recs,
    }

    # Write output JSON BEFORE gating/verifier
    tmp_out = out_json + ".tmp"
    with open(tmp_out, "w") as f:
        json.dump(final_payload, f, indent=1)
    os.replace(tmp_out, out_json)
    print(f"[+] Final C24 artifact saved to: {out_json} ({os.path.getsize(out_json)} bytes)")

    # Record status JSON
    import hashlib
    with open(out_json, "rb") as f:
        artifact_sha = hashlib.sha256(f.read()).hexdigest()

    tag = f"{tag}_smoke" if smoke else tag
    status_path = os.path.join(REPO_ROOT, "results/campaigns/status", f"{tag}.json")
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    out_rel = out_json.replace(REPO_ROOT + "/", "")
    status_payload = {
        "campaign": tag,
        "id": tag,
        "state": "done",
        "provenance": {
            "commit": manifest.get("git_commit"),
            "dirty": manifest.get("git_dirty"),
        },
        "cuda_available": False,
        "wall_clock_min": total_wall / 60.0,
        "outputs": {
            out_rel: {
                "sha256": artifact_sha,
                "bytes": os.path.getsize(out_json),
            }
        },
    }
    with open(status_path, "w") as f:
        json.dump(status_payload, f, indent=2)

    return final_payload


def main():
    parser = argparse.ArgumentParser(description="C24 Adaptive Access Sweeps Runner")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (1 seed, 2 rounds)")
    parser.add_argument("--pilot", action="store_true", help="Run pilot timing (seed 2000, 40 rounds, all cells)")
    parser.add_argument("--workers", type=int, default=6, help="Worker processes for multiprocessing")
    parser.add_argument("--out-json", type=str, default="results/campaigns/C24_adaptive_access.json")
    parser.add_argument("--schedules", nargs="+", default=None, help="Schedules to run (S1, S2, S3)")
    parser.add_argument("--include-f04", action="store_true", help="Include f/K=0.4 sweep if budget permits")
    parser.add_argument("--methods", nargs="+", default=None, help="Subset of methods to run")
    parser.add_argument("--no-benign", action="store_true", help="Skip the benign baseline cells")
    parser.add_argument("--tag", default="C24", help="Campaign ID used for the status file")
    args = parser.parse_args()

    if args.pilot:
        run_pilot(workers=args.workers)
    else:
        out = "results/campaigns/smoke/C24_smoke.json" if args.smoke else args.out_json
        run_campaign(
            out_json=out,
            schedules=args.schedules,
            include_f04=args.include_f04,
            smoke=args.smoke,
            workers=args.workers,
            methods=args.methods,
            include_benign=not args.no_benign,
            tag=args.tag,
        )


if __name__ == "__main__":
    main()
