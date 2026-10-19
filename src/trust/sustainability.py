"""Sustainability and efficiency accounting.

Real-world FL deployment is constrained by communication and energy, not just
accuracy. We report model size, per-round and total communication volume, a
FLOP estimate for a forward pass, and a transparent wall-clock-based energy
proxy. TrustFedGNN transmits only model weights (the FTGD statistics add O(1)
scalars), so its communication footprint equals FedAvg's -- an important
"fairness/privacy at no extra communication cost" point.
"""
from __future__ import annotations

from typing import Dict

import torch

# very rough CPU power draw for the energy proxy (documented as indicative only)
_CPU_WATTS = 65.0


def model_size(model) -> Dict[str, float]:
    n = sum(p.numel() for p in model.parameters())
    bytes4 = n * 4
    return {"params": int(n), "size_mb": round(bytes4 / 1e6, 4)}


def communication_cost(num_params: int, rounds: int, num_clients: int,
                       participation: float = 1.0) -> Dict[str, float]:
    """Total bytes exchanged (float32, upload+download each round)."""
    active = max(1, int(round(num_clients * participation)))
    per_round = 2 * active * num_params * 4          # up + down
    total = per_round * rounds
    return {
        "params": int(num_params),
        "per_round_mb": round(per_round / 1e6, 3),
        "total_gb": round(total / 1e9, 4),
        "rounds": rounds, "active_clients": active,
    }


def forward_flops(model, data) -> Dict[str, float]:
    """Coarse forward-pass FLOP estimate: dense layers (2*in*out per node) plus
    message passing (2*E*hidden). Indicative, not exact."""
    n = int(data.num_nodes)
    e = int(data.edge_index.shape[1])
    h = getattr(model, "hidden_channels", 64)
    lin = sum(p.numel() for p in model.parameters()) * 2 * n
    mp = 2 * e * h * getattr(model, "num_layers", 2)
    return {"approx_gflops_fwd": round((lin + mp) / 1e9, 4)}


def energy_proxy(wall_seconds: float) -> Dict[str, float]:
    """Indicative energy from wall-clock x assumed CPU power. A *proxy* for
    relative comparison between methods on the same hardware -- not a certified
    measurement."""
    joules = wall_seconds * _CPU_WATTS
    return {"wall_s": round(wall_seconds, 1),
            "energy_wh": round(joules / 3600.0, 3),
            "co2e_g_approx": round(joules / 3600.0 * 0.4, 3)}  # ~0.4 gCO2/Wh grid avg


def sustainability_report(model, data, config, wall_seconds: float = 0.0) -> Dict:
    ms = model_size(model)
    cc = communication_cost(ms["params"], config.rounds, config.num_clients)
    out = {**ms, **cc, **forward_flops(model, data)}
    if wall_seconds:
        out.update(energy_proxy(wall_seconds))
    return out
