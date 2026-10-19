"""Regression test for FSER beta_init parameter wiring and gradient flow.

Guards against B-01 bug where beta_init in ExperimentConfig was ignored and
hardcoded to 0.5.
"""
import pytest
import torch
import torch.nn.functional as F

from src.config import ExperimentConfig
from src.models import build_model
from src.models.gnn import FSERLayer, TrustFedGNN


def test_fser_layer_beta_init():
    """Verify FSERLayer initializes beta correctly and preserves it across resets."""
    for b in [0.0, 0.1, 0.3, 0.8, 2.5]:
        layer = FSERLayer(in_channels=16, out_channels=4, heads=2, beta_init=b)
        assert layer.beta.item() == pytest.approx(b, rel=1e-5), f"Expected beta={b}, got {layer.beta.item()}"
        
        # Test reset_parameters preserves configured beta_init
        layer.reset_parameters()
        assert layer.beta.item() == pytest.approx(b, rel=1e-5), f"Expected beta={b} after reset, got {layer.beta.item()}"


def test_build_model_trustfedgnn_beta_init():
    """Verify build_model forwards config.beta_init to all FSER layers."""
    for b in [0.05, 0.25, 0.75, 1.5]:
        cfg = ExperimentConfig(model="trustfedgnn", beta_init=b, num_layers=3, heads=2)
        model = build_model("trustfedgnn", in_channels=10, config=cfg)
        assert isinstance(model, TrustFedGNN)
        assert len(model.layers) == 3
        for i, layer in enumerate(model.layers):
            assert layer.beta.item() == pytest.approx(b, rel=1e-5), (
                f"Layer {i} beta expected {b}, got {layer.beta.item()}"
            )


def test_fser_beta_receives_gradient():
    """Verify that self.beta receives non-zero gradients across all FSER modes."""
    from src.data.datasets import load_synthetic
    data = load_synthetic(seed=42, num_nodes=50)
    for mode in ["sub", "add", "same_penalize"]:
        torch.manual_seed(42)
        cfg = ExperimentConfig(model="trustfedgnn", beta_init=0.5, fser_mode=mode, num_layers=1, heads=2, dropout=0.0)
        model = build_model("trustfedgnn", in_channels=data.x.shape[1], config=cfg)
        model.train()
        
        out = model(data.x, data.edge_index, sensitive_attr=data.sensitive_attr)
        loss = out[data.train_mask].sum()
        loss.backward()
        
        beta_grad = model.layers[0].beta.grad
        assert beta_grad is not None, f"Mode '{mode}': beta.grad should not be None"
        assert beta_grad.abs().item() > 1e-6, f"Mode '{mode}': beta.grad should be non-zero, got {beta_grad.item()}"




def test_fser_modes():
    """Verify FSERLayer supports sub, add, and same_penalize modes."""
    for mode in ["sub", "add", "same_penalize"]:
        cfg = ExperimentConfig(model="trustfedgnn", beta_init=0.5, fser_mode=mode)
        model = build_model("trustfedgnn", in_channels=8, config=cfg)
        assert model.fser_mode == mode
        assert model.layers[0].fser_mode == mode


def test_fser_mode_defaults_agree():
    """The default fser_mode must be the canonical 'sub' at EVERY construction site.

    A direct ``TrustFedGNN(...)`` call bypasses build_model/ExperimentConfig, so a
    divergent constructor default would silently train a different method with no
    error -- the layer, the backbone and the canonical config must all say 'sub'.
    """
    assert FSERLayer(in_channels=8, out_channels=4, heads=2).fser_mode == "sub"
    backbone = TrustFedGNN(in_channels=8, hidden_channels=8, heads=2, num_layers=1)
    assert backbone.fser_mode == "sub"
    assert backbone.layers[0].fser_mode == "sub"
    assert ExperimentConfig().fser_mode == "sub"
    assert ExperimentConfig.canonical().fser_mode == "sub"


def test_canonical_matches_manifest():
    """Verify ExperimentConfig.canonical() initializes with the frozen canonical hyperparameters."""
    cfg = ExperimentConfig.canonical(seed=42)
    assert cfg.fser_mode == "sub", f"Expected sub, got {cfg.fser_mode}"
    assert cfg.dp_enabled is True
    assert cfg.dp_mode == "ftgd"
    assert cfg.model == "trustfedgnn"
    assert cfg.aggregator == "fu_shapley"
    assert cfg.fu_alpha == pytest.approx(0.1)
    assert cfg.fu_ema_beta == pytest.approx(0.9)



# --------------------- freeze_beta: the true FSER ablation --------------------- #
def test_freeze_beta_makes_beta_non_trainable():
    """D2(a). ``ours-nofser`` swaps model='gat', which removes FSER *and* the
    whole TrustFedGNN scaffold (BatchNorm, residual, skip-concat) in one move --
    so any gain it shows is attributed to FSER when part of it is architecture.
    freeze_beta holds beta at beta_init inside the unchanged backbone."""
    cfg = ExperimentConfig(model="trustfedgnn", beta_init=0.0, freeze_beta=True,
                           num_layers=2, heads=2, dropout=0.0)
    model = build_model("trustfedgnn", in_channels=8, config=cfg)

    ids = {id(p) for p in model.parameters()}
    for layer in model.layers:
        assert layer.beta.requires_grad is False, "frozen beta must not require grad"
        assert id(layer.beta) not in ids, "frozen beta must be outside parameters()"
        assert float(layer.beta) == pytest.approx(0.0)
    # it keeps its slot in state_dict, so the federated flat vector is unchanged
    assert "layers.0.beta" in model.state_dict()

    # ... and stays frozen through a backward pass
    from src.data.datasets import load_synthetic
    d = load_synthetic(seed=0, num_nodes=40)
    m2 = build_model("trustfedgnn", in_channels=d.x.shape[1], config=cfg)
    m2.train()
    m2(d.x, d.edge_index, d.sensitive_attr).sum().backward()
    for layer in m2.layers:
        assert layer.beta.grad is None, "no gradient may reach a frozen beta"
    assert m2.layers[0].lin.weight.grad is not None, "the rest still trains"

    m2.clamp_beta()                                  # must not disturb the constant
    assert float(m2.layers[0].beta) == pytest.approx(0.0)

    # unfrozen is the default and stays a trainable Parameter
    hot = build_model("trustfedgnn", in_channels=8,
                      config=ExperimentConfig(model="trustfedgnn", beta_init=0.0,
                                              num_layers=1, heads=2))
    assert hot.layers[0].beta.requires_grad is True


def test_frozen_zero_beta_is_exactly_plain_gat_attention():
    """D2(b), the load-bearing claim: at beta=0 the FSER logit correction is
    annihilated for every mode, so FSERLayer computes *exactly* GAT attention.
    Compared against the plain-GAT formula recomputed by hand from the layer's
    own lin/att weights -- not against src.models.GAT, whose backbone differs."""
    from torch_geometric.utils import softmax as pyg_softmax

    torch.manual_seed(0)
    N, F_in, H, C = 12, 7, 3, 5
    x = torch.randn(N, F_in)
    edge_index = torch.stack([
        torch.randint(0, N, (40,)), torch.randint(0, N, (40,))])
    s = (torch.arange(N) % 2)                        # a genuinely mixed graph

    layer = FSERLayer(F_in, C, heads=H, concat=True, dropout=0.0,
                      beta_init=0.0, freeze_beta=True)
    layer.eval()
    got = layer(x, edge_index, s)

    # plain GAT: e_ij = leakyrelu([Wh_i || Wh_j] . a), alpha = softmax_i(e_ij)
    h = layer.lin(x).view(N, H, C)
    src_, dst_ = edge_index[0], edge_index[1]        # PyG flow: x_j = x[src]
    x_i, x_j = h[dst_], h[src_]
    e = torch.nn.functional.leaky_relu(
        (torch.cat([x_i, x_j], dim=-1) * layer.att).sum(-1), 0.2)
    alpha = pyg_softmax(e, dst_, num_nodes=N)
    want = torch.zeros(N, H, C).index_add_(0, dst_, x_j * alpha.unsqueeze(-1))
    want = want.reshape(N, H * C)

    assert torch.allclose(got, want, atol=1e-6), \
        "FSERLayer at beta=0 must be bit-for-bit plain GAT attention"

    # and the mode is irrelevant once beta is pinned at zero
    for mode in ("sub", "add", "same_penalize"):
        alt = FSERLayer(F_in, C, heads=H, concat=True, dropout=0.0,
                        beta_init=0.0, freeze_beta=True, fser_mode=mode)
        alt.load_state_dict(layer.state_dict())
        alt.eval()
        assert torch.allclose(alt(x, edge_index, s), got, atol=1e-6), \
            f"mode {mode!r} must be indistinguishable at beta=0"


def test_missing_sensitive_attr_warns_instead_of_silently_degenerating():
    """D3. ``sensitive_attr=None`` zero-fills s. That is not neutral: for
    sub/add every pair compares equal so phi == 0 and the layer quietly becomes
    plain GAT, while same_penalize instead applies the FULL penalty to every
    edge. Both used to happen in silence."""
    import warnings as _w

    x = torch.randn(6, 8)
    ei = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])

    for mode, needle in (("sub", "degenerates to plain GAT"),
                         ("add", "degenerates to plain GAT"),
                         ("same_penalize", "EVERY edge")):
        layer = FSERLayer(8, 4, heads=2, dropout=0.0, fser_mode=mode)
        with pytest.warns(RuntimeWarning, match="sensitive_attr=None") as rec:
            layer(x, ei, None)
        assert any(needle in str(r.message) for r in rec), \
            f"mode {mode!r} must say what the zero-fill does *for that mode*"
        assert any(repr(mode) in str(r.message) for r in rec)

    # passing a real s stays silent
    layer = FSERLayer(8, 4, heads=2, dropout=0.0)
    with _w.catch_warnings():
        _w.simplefilter("error")
        layer(x, ei, torch.tensor([0, 1, 0, 1, 0, 1]))
