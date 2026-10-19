"""Pure-PyTorch multi-hop neighbor sampler.

A dependency-free replacement for ``torch_geometric.loader.NeighborLoader``
(which requires the compiled ``pyg-lib``/``torch-sparse`` extensions that are
not available for every torch/Python combination, e.g. torch 2.11 + Py3.12 on
Colab). GraphSAGE-style fan-out sampling with replacement, fully vectorised, so
it scales to millions of nodes/edges on CPU or GPU.

Each yielded batch is a :class:`~torch_geometric.data.Data` object whose first
``batch_size`` rows are the seed nodes (matching NeighborLoader's convention),
with the induced sampled subgraph relabelled to local indices.
"""
from __future__ import annotations

import torch
from torch_geometric.data import Data


class SimpleNeighborLoader:
    def __init__(self, data, num_neighbors, batch_size, input_nodes, shuffle=True):
        self.data = data
        self.fanout = list(num_neighbors)
        self.bs = batch_size
        self.shuffle = shuffle
        if input_nodes.dtype == torch.bool:
            input_nodes = input_nodes.nonzero(as_tuple=True)[0]
        self.seeds = input_nodes
        # CSR over the (already undirected) edge_index, sorted by source.
        # All CSR tensors live on the graph's device so indexing by GPU node
        # ids is valid.
        ei = data.edge_index
        self.device = ei.device
        n = data.num_nodes
        self.n = n
        order = torch.argsort(ei[0])
        self.col = ei[1][order].contiguous()
        deg = torch.bincount(ei[0], minlength=n)
        self.rowptr = torch.zeros(n + 1, dtype=torch.long, device=self.device)
        self.rowptr[1:] = torch.cumsum(deg, 0)
        self.deg = deg
        self.seeds = self.seeds.to(self.device)

    def __len__(self):
        return (len(self.seeds) + self.bs - 1) // self.bs

    def __iter__(self):
        seeds = self.seeds
        if self.shuffle:
            seeds = seeds[torch.randperm(len(seeds))]
        for i in range(0, len(seeds), self.bs):
            yield self._sample(seeds[i:i + self.bs])

    def _sample_neighbors(self, nodes, k):
        """Sample up to k neighbours (with replacement) for each node -> (src, dst)."""
        deg = self.deg[nodes]
        has = deg > 0
        nodes = nodes[has]; deg = deg[has]
        if len(nodes) == 0:
            return nodes.new_empty(0), nodes.new_empty(0)
        rand = (torch.rand(len(nodes), k, device=nodes.device) * deg.unsqueeze(1)).long()
        pos = self.rowptr[nodes].unsqueeze(1) + rand              # [m, k]
        dst = self.col[pos.reshape(-1)]                            # sampled neighbours
        src = nodes.repeat_interleave(k)
        return src, dst

    def _sample(self, seeds):
        device = self.device
        seeds = seeds.to(device)
        node_list = [seeds]
        src_all, dst_all = [], []
        frontier = seeds
        for k in self.fanout:
            src, dst = self._sample_neighbors(frontier, k)
            if len(src):
                src_all.append(src); dst_all.append(dst)
                node_list.append(dst)
            frontier = torch.unique(dst) if len(dst) else frontier
        nodes = torch.unique(torch.cat(node_list))
        # ensure seeds occupy the first rows (NeighborLoader convention)
        seed_set = seeds
        rest = nodes[~torch.isin(nodes, seed_set)]
        nodes = torch.cat([seed_set, rest])
        # global -> local remap
        remap = torch.full((int(self.rowptr.numel() - 1),), -1, dtype=torch.long, device=device)
        remap[nodes] = torch.arange(len(nodes), device=device)
        if src_all:
            src = remap[torch.cat(src_all)]; dst = remap[torch.cat(dst_all)]
            edge_index = torch.stack([src, dst])
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        d = self.data
        b = Data(x=d.x[nodes], edge_index=edge_index, y=d.y[nodes],
                 sensitive_attr=d.sensitive_attr[nodes])
        b.batch_size = len(seeds)
        return b
