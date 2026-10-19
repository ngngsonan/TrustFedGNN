"""Model factory."""
from __future__ import annotations

from .gnn import TrustFedGNN, GCN, GAT, FairGNN, FairSIN, FaVGNN, FSERLayer

_REGISTRY = {
    "trustfedgnn": TrustFedGNN,
    "gcn": GCN,
    "gat": GAT,
    "fairgnn": FairGNN,
    "fairsin": FairSIN,
    "favgnn": FaVGNN,
}


def build_model(name: str, in_channels: int, config=None):
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_REGISTRY)}")
    kwargs = dict(in_channels=in_channels, out_channels=1)
    if config is not None:
        kwargs.update(
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
            heads=config.heads,
            dropout=config.dropout,
            beta_init=getattr(config, "beta_init", 0.5),
            fser_mode=getattr(config, "fser_mode", "sub"),
            freeze_beta=getattr(config, "freeze_beta", False),
        )
    return _REGISTRY[name](**kwargs)


__all__ = ["build_model", "TrustFedGNN", "GCN", "GAT", "FairGNN", "FairSIN", "FaVGNN", "FSERLayer"]
