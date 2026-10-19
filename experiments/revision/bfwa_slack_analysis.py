"""BFWA disparity-constraint slack under DP noise.

Measures the gap between the disparity clients *report* once it is privatised and
the disparity that is actually there:
    Slack_t = | sum_k w_k * DPD_released_k - sum_k w_k * DPD_true_k |
against the fairness budget tau. This is the empirical side of the LDP-barrier
result in the manuscript: once the report is privatised, the constraint stops tracking fairness.

CORRECTED 11-09-2026 (Phase 2a). The previous version re-implemented the noise
mechanism here and got it wrong three ways, inflating the published slack by
roughly 2.2-2.5x:

  1. it used ``sqrt(2/n)`` -- a standard error, O(1/sqrt(n)) -- where the
     deployed mechanism uses the L2 sensitivity ``sqrt(1/n0^2 + 1/n1^2)``,
     O(1/n), the quantity assumption A4 actually bounds;
  2. it read ``len(train_mask)`` (the mask's length, i.e. the whole subgraph)
     instead of ``train_mask.sum()`` (the number of training nodes);
  3. it added one noise draw to the scalar DPD and clipped at zero, whereas the
     mechanism noises mu0 and mu1 independently and then folds |mu0 - mu1| --
     a different distribution with a different, systematically positive bias.

The fix is not a better re-implementation: it calls the production path
(``Client._released_disparity``) so the measurement cannot drift from the
mechanism again.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch

from src.config import ExperimentConfig
from src.federated.trainer import FederatedTrainer
from src.federated.client import _soft_dpd
from src.utils.provenance import build_manifest




def folded_normal_mean(delta: float, sigma_tilde: float) -> float:
    """Exact E[|N(delta, sigma_tilde^2)|] (folded-normal mean).

        E[DPD~_k] = sigma~_k sqrt(2/pi) exp(-delta_k^2 / 2 sigma~_k^2)
                    + delta_k erf(delta_k / (sigma~_k sqrt 2))

    The limit sigma~ sqrt(2/pi) is what Theorem 4 uses, and it holds only when
    sigma~ >> |delta|. Testing the measurement against that LIMIT is close to
    circular: it can only agree in the regime whose assumption it encodes, and a
    referee is entitled to say so. The exact expression has no such regime
    condition, so it is what the data should be compared against.

    Even in delta, so the unsigned per-client disparity is sufficient.
    """
    if sigma_tilde <= 0:
        return abs(delta)
    d = abs(delta)
    return (sigma_tilde * math.sqrt(2.0 / math.pi) * math.exp(-(d * d) / (2.0 * sigma_tilde * sigma_tilde))
            + d * math.erf(d / (sigma_tilde * math.sqrt(2.0))))


def _pair_theorem4(per_round):
    """Summarise Theorem 4 round by round, not as a ratio of means.

    Two forms are checked separately because they say different things:

      equality form  E[sum_k w_k DPD~_k] = sqrt(2/pi) * sum_k w_k sigma~_k,
                     valid in the noise-dominant regime -> ratio should sit
                     near 1, and drift BELOW 1 marks rounds where the true
                     disparity is not yet negligible against the noise.

      lower bound    E[slack] >= sqrt(2/pi) * sum_k w_k sigma~_k - sum_k w_k|delta_k|,
                     so slack/predicted >= 1 is the claim; rounds below 1 are
                     where the delta term still bites and must be reported, not
                     averaged away.
    """
    if not per_round:
        return None
    def _boot_ci(a, n=10000):
        rng = np.random.default_rng(0)
        b = [rng.choice(a, size=len(a), replace=True).mean() for _ in range(n)]
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

    r = np.array([x["ratio_observed_over_predicted"] for x in per_round], float)
    sl = np.array([x["slack_over_predicted"] for x in per_round], float)
    ex = np.array([x["ratio_observed_over_exact"] for x in per_round
                   if x.get("ratio_observed_over_exact") is not None], float)
    return {
        "n_rounds": int(len(per_round)),
        # The comparison that carries no regime assumption. This is the one to
        # read; the asymptotic ratio below is kept to show WHERE the
        # noise-dominant approximation stops holding.
        "ratio_observed_over_exact": ({
            "mean": float(ex.mean()), "ci95_of_mean": _boot_ci(ex),
            "median": float(np.median(ex)),
            "min": float(ex.min()), "max": float(ex.max()),
            "frac_within_10pct_of_1": float(np.mean(np.abs(ex - 1.0) <= 0.10)),
        } if len(ex) else None),
        # Theorem 4 constrains an EXPECTATION, so the quantity to read is the
        # mean with its interval. A single round is a folded-normal draw whose
        # mass sits near zero, so individual rounds below 1 are what the
        # distribution looks like, not a refutation -- hence min/max and the
        # fraction are reported for shape, never as a verdict.
        "ratio_observed_over_predicted": {
            "mean": float(r.mean()), "ci95_of_mean": _boot_ci(r),
            "median": float(np.median(r)),
            "min": float(r.min()), "max": float(r.max()),
            "frac_within_10pct_of_1": float(np.mean(np.abs(r - 1.0) <= 0.10)),
        },
        "slack_over_predicted": {
            "mean": float(sl.mean()), "ci95_of_mean": _boot_ci(sl),
            "median": float(np.median(sl)),
            "min": float(sl.min()), "max": float(sl.max()),
            "frac_at_or_above_1": float(np.mean(sl >= 1.0)),
        },
    }


def analyze_bfwa_slack(dataset="german", seeds=tuple(range(42, 52)),
                       epsilons=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
                       rounds=20, num_clients=10, tau=0.05,
                       log_sigma_vector: bool = False):
    summary_by_eps = {}

    for eps in epsilons:
        slacks_all_rounds = []
        sigma_tilde_weighted = []
        true_dpd_aggregates = []
        noisy_dpd_aggregates = []
        # Theorem 4 is a statement about ONE release: the observed aggregate of a
        # single round against the sigma of that same round. Comparing
        # mean(observed) with mean(sigma) instead is a ratio of means, and the
        # weights w vary round to round, so the two disagree for reasons that
        # have nothing to do with the theorem. Pair them per round.
        per_round = []

        for s in seeds:
            cfg = ExperimentConfig.canonical(
                dataset=dataset, seed=s, rounds=rounds, num_clients=num_clients,
                model="trustfedgnn", aggregator="bfwa", fairness_budget=tau,
                local_fairness=True, dp_enabled=True, dp_epsilon=eps, dp_delta=1e-5
            )
            trainer = FederatedTrainer(cfg)

            for t in range(rounds):
                # Run round
                rec = trainer._round(t)
                w = rec.get("agg_weights")
                if w is None or len(w) != num_clients:
                    w = [1.0 / num_clients] * num_clients
                w = np.array(w)

                # Compute true and reported DPD for each client
                dpd_true_list = []
                dpd_noisy_list = []
                sigma_tilde_list = []      # sqrt(2)*z*Delta_mu, per client
                client_vectors = []

                for c_idx, c in enumerate(trainer.clients):
                    c.model.eval()
                    d = c.data
                    sel = d.train_mask
                    s_sel = d.sensitive_attr[sel]
                    with torch.no_grad():
                        pred_sel = c.model(d.x, d.edge_index, d.sensitive_attr)[sel]

                        # TRUE: same release pass, no noise. _release_pred is the
                        # s-blind forward when the mechanism is live, so true and
                        # released are compared on the same predictions.
                        pred_rel = c._release_pred(pred_sel, d.x, d.edge_index, sel)
                        true_val = float(_soft_dpd(pred_rel, s_sel))

                        # RELEASED: production mechanism -- correct sensitivity,
                        # independent noise on mu0 and mu1, then the fold.
                        c._last_privatised_dpd = None
                        c._released_disparity(pred_sel, s_sel, d.x, d.edge_index, sel)
                        noisy_val = c._last_privatised_dpd
                        if noisy_val is None:          # sigma == 0: nothing released
                            noisy_val = true_val

                        n0 = int((s_sel == 0).sum()); n1 = int((s_sel == 1).sum())
                        if n0 > 0 and n1 > 0:
                            delta_mu = (1.0 / n0 ** 2 + 1.0 / n1 ** 2) ** 0.5
                            st_val = float(math.sqrt(2.0) * c.noise_multiplier * delta_mu)
                            sigma_tilde_list.append(st_val)
                        else:
                            st_val = None

                    dpd_true_list.append(true_val)
                    dpd_noisy_list.append(noisy_val)
                    if log_sigma_vector:
                        client_vectors.append({
                            "client_id": c_idx,
                            "n0": n0,
                            "n1": n1,
                            "sigma_tilde": st_val,
                            "w": float(w[c_idx]),
                            "delta_true": true_val,
                            "released": float(noisy_val),
                        })

                dpd_true_arr = np.array(dpd_true_list)
                dpd_noisy_arr = np.array(dpd_noisy_list)

                agg_true = float(np.sum(w * dpd_true_arr))
                agg_noisy = float(np.sum(w * dpd_noisy_arr))
                slack = abs(agg_noisy - agg_true)

                slacks_all_rounds.append(slack)
                sigma_w = (float(np.sum(w * np.array(sigma_tilde_list)))
                           if sigma_tilde_list else None)
                if sigma_w is not None:
                    sigma_tilde_weighted.append(sigma_w)
                true_dpd_aggregates.append(agg_true)
                noisy_dpd_aggregates.append(agg_noisy)

                if log_sigma_vector and sigma_w is not None:
                    sigma_w_assert = float(sum(cv["w"] * cv["sigma_tilde"]
                                               for cv in client_vectors if cv["sigma_tilde"] is not None))
                    assert math.isclose(sigma_w, sigma_w_assert, rel_tol=1e-5), (
                        f"sigma_w mismatch: {sigma_w} vs {sigma_w_assert}"
                    )

                if sigma_w:
                    predicted = math.sqrt(2.0 / math.pi) * sigma_w
                    # Exact Lemma 4.1 expectation, aggregated with the same
                    # weights. Unlike the asymptotic form this carries no regime
                    # condition, so it is the honest target for the measurement.
                    predicted_exact = float(np.sum(w * np.array(
                        [folded_normal_mean(dt, st)
                         for dt, st in zip(dpd_true_list, sigma_tilde_list)])))
                    rec_round = {
                        "seed": int(s), "round": int(t),
                        "agg_true": agg_true, "agg_noisy": agg_noisy,
                        "slack": slack, "sigma_w": sigma_w,
                        "predicted_noisy_agg": predicted,
                        "predicted_noisy_agg_exact": predicted_exact,
                        "ratio_observed_over_exact": (agg_noisy / predicted_exact
                                                      if predicted_exact else None),
                        # equality form of Theorem 4, in the noise-dominant regime
                        "ratio_observed_over_predicted": agg_noisy / predicted,
                        # lower-bound form: E[slack] >= sqrt(2/pi) * sigma_w - sum_k w_k|delta_k|
                        "slack_over_predicted": slack / predicted,
                    }
                    if log_sigma_vector:
                        rec_round["client_vectors"] = client_vectors
                    per_round.append(rec_round)

        mean_slack = float(np.mean(slacks_all_rounds))
        std_slack = float(np.std(slacks_all_rounds))
        slack_ratio = mean_slack / tau  # fraction of tau

        summary_by_eps[str(eps)] = {
            "mean_slack": mean_slack,
            "std_slack": std_slack,
            "slack_ratio_of_tau": slack_ratio,
            "mean_true_dpd_agg": float(np.mean(true_dpd_aggregates)),
            "mean_noisy_dpd_agg": float(np.mean(noisy_dpd_aggregates)),
            "tau": tau,
            # Folded-normal bias: E[released aggregate] = sqrt(2/pi) * sum_k w_k * sigma_tilde_k
            "mean_sigma_tilde_weighted": float(np.mean(sigma_tilde_weighted)) if sigma_tilde_weighted else None,
            "theorem4_predicted_noisy_agg": (float(np.mean(sigma_tilde_weighted)) * math.sqrt(2.0 / math.pi))
                                            if sigma_tilde_weighted else None,
            "theorem4_paired": _pair_theorem4(per_round),
            "per_round": per_round,
        }
        print(f"[*] eps={eps} -> Slack = {mean_slack:.4f} +/- {std_slack:.4f} ({slack_ratio * 100:.1f}% of tau={tau})", flush=True)

    return summary_by_eps



# --------------------------------------------------------------------------- #
# Caption text, derived from the measurement
# --------------------------------------------------------------------------- #
# The caption used to assert "a negligible slack of $<0.003$ ($<6\%$ of $\tau$)"
# as a fixed string, printed verbatim whatever the run produced -- i.e. a
# conclusion written before the experiment. Everything below is computed from
# the results dict instead. Cut-offs, stated so a reader can disagree with them
# rather than having to reverse-engineer them:
#
#   slack / tau  <  10%  -> "negligible"  (well inside the budget's own slack)
#              <  50%  -> "moderate"    (eats a noticeable share of the budget)
#              >= 50%  -> "substantial" (the reported constraint is no longer
#                                        a reliable proxy for the true one)
SLACK_BUCKETS = ((10.0, "negligible"), (50.0, "moderate"), (float("inf"), "substantial"))


def slack_bucket(pct_of_tau: float) -> str:
    """Qualitative label for a measured slack, as a percentage of tau."""
    for hi, name in SLACK_BUCKETS:
        if pct_of_tau < hi:
            return name
    return "substantial"                                  # pragma: no cover


def _slack_caption_sentence(results: dict, tau: float) -> str:
    """Sentence describing the slack at the *largest* epsilon tested.

    The largest epsilon is the weakest-privacy / lowest-noise setting, i.e. the
    deployed operating point and the most favourable case for the method; if
    the slack is not negligible there it is not negligible anywhere.
    """
    clean_results = {k: v for k, v in results.items() if k not in ("manifest", "_manifest")}
    if not clean_results:
        return "No slack measurements were produced by this run."
    eps_key = max(clean_results, key=lambda k: float(k))
    v = clean_results[eps_key]
    pct = 100.0 * v["slack_ratio_of_tau"]
    word = slack_bucket(pct)
    if word == "negligible":
        implication = ("so FTGD privacy does not destabilise the BFWA dual "
                       "iteration at this operating point")
    elif word == "moderate":
        implication = ("so the reported constraint value carries a non-trivial "
                       "DP-induced error that the budget $\\tau$ must absorb")
    else:
        implication = ("so at this noise level the reported disparity is no longer "
                       "a reliable stand-in for the true one, and $\\tau$ must be "
                       "tightened (or the privacy budget loosened) for the "
                       "constraint to mean what it says")
    return (f"At the weakest-privacy point tested ($\\epsilon = {eps_key}$), DP noise "
            f"induces a {word} slack of ${v['mean_slack']:.4f} \\pm "
            f"{v['std_slack']:.4f}$ (${pct:.1f}\\%$ of $\\tau = {tau}$), "
            f"{implication}.")


def run_bfwa_slack_experiment(out_json="results/revision/bfwa_slack.json",
                              out_tex="build/tables/bfwa_slack.tex",
                              dataset="bail", seeds=(42, 43),
                              epsilons=(2.0, 4.0, 8.0), rounds=20,
                              num_clients=10, tau=0.05,
                              log_sigma_vector: bool = False):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)

    print(f"[*] Running BFWA DP-induced disparity slack analysis on {dataset}...", flush=True)
    results = analyze_bfwa_slack(dataset=dataset, seeds=seeds, epsilons=epsilons,
                                 rounds=rounds, num_clients=num_clients, tau=tau,
                                 log_sigma_vector=log_sigma_vector)
    for k, v in list(results.items()):
        if k in ("manifest", "_manifest"):
            continue
        v["slack_bucket"] = slack_bucket(100.0 * v["slack_ratio_of_tau"])

    results["manifest"] = build_manifest(
        experiment="bfwa_constraint_slack",
        args={
            "dataset": dataset,
            "seeds": list(seeds),
            "epsilons": list(epsilons),
            "rounds": rounds,
            "num_clients": num_clients,
            "tau": tau,
            "log_sigma_vector": log_sigma_vector,
            "note": ("Noise comes from Client._released_disparity, the production path, "
                     "not from a local re-implementation. See the module docstring for the "
                     "three defects that re-implementation had."),
        },
    )

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved BFWA slack JSON to {out_json}")

    # Generate LaTeX table
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{\\textbf{BFWA Disparity Constraint Slack Induced by Differential Privacy Noise.}",
        "Empirical slack between reported noisy disparity and true underlying disparity "
        "$\\Delta_{\\tau} = |\\sum_k w_k \\widehat{\\text{DPD}}_k - \\sum_k w_k \\text{DPD}_k|$ "
        f"under budget $\\tau = {tau}$ ({dataset.capitalize()}, $n={num_clients}$ clients, "
        f"{rounds} rounds, {len(seeds)} seed" + ("s" if len(seeds) != 1 else "") + ").",
        _slack_caption_sentence(results, tau) + "}",
        "\\label{tab:bfwa_slack}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{DP Target $\\epsilon$} & \\textbf{Reported $\\sum w_k \\widehat{\\text{DPD}}_k$} & \\textbf{True $\\sum w_k \\text{DPD}_k$} & \\textbf{Slack $|\\Delta_{\\tau}|$} & \\textbf{Slack / $\\tau$ (\\%)} \\\\",
        "\\midrule",
    ]

    for eps_str, v in results.items():
        if eps_str in ("manifest", "_manifest"):
            continue
        line = (
            f"$\\epsilon = {eps_str}$ & {v['mean_noisy_dpd_agg']:.4f} & {v['mean_true_dpd_agg']:.4f} & "
            f"{v['mean_slack']:.4f} $\\pm$ {v['std_slack']:.4f} & {v['slack_ratio_of_tau'] * 100:.1f}\\% \\\\"
        )
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] Saved LaTeX BFWA slack table to {out_tex}")


def main():
    ap = argparse.ArgumentParser(description="BFWA DP-induced constraint slack.")
    ap.add_argument("--dataset", default="german")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--epsilons", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--num-clients", type=int, default=10)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--log-sigma-vector", action="store_true", default=False,
                    help="Record per-client noise and disparity vectors (K12/C21)")
    ap.add_argument("--out-json", default="results/revision/bfwa_slack.json")
    ap.add_argument("--out-tex", default="build/tables/bfwa_slack.tex")
    a = ap.parse_args()
    run_bfwa_slack_experiment(out_json=a.out_json, out_tex=a.out_tex,
                              dataset=a.dataset, seeds=tuple(a.seeds),
                              epsilons=tuple(a.epsilons), rounds=a.rounds,
                              num_clients=a.num_clients, tau=a.tau,
                              log_sigma_vector=a.log_sigma_vector)


if __name__ == "__main__":
    main()
