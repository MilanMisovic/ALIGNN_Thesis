"""
Dataset wrapper that attaches precomputed bipartite pore graphs to ALIGNN
batches. Mirrors the PoreAugmentedDataset pattern from pore_utils.py exactly.

Usage (in train.py, replacing get_pore_loaders):
    from pore_graph_dataset import get_pore_graph_loaders
    train_loader, val_loader, test_loader, prepare_batch = get_pore_graph_loaders(config)

The batch then yields:
    (g, lg, lat, G_pore, target)
instead of:
    (g, lg, lat, target)

where G_pore is a batched DGL graph (or zero graph for
structures with no accessible pores).
"""

import torch
import dgl
import numpy as np
from torch.utils.data import Dataset
from dgl.dataloading import GraphDataLoader


# Fallback graph for structures with no accessible pores
# A minimal valid heterograph with 1 atom node, 1 pore node, 0 edges.
# The GNN branch will produce a zero embedding for these structures,
# which is the correct behaviour, no pore information available.

def _make_empty_pore_graph(n_atom_features=92):
    """Return a minimal valid bipartite graph with no edges."""
    G = dgl.heterograph(
        {
            ('pore', 'wall',     'atom'): ([], []),
            ('atom', 'wall_rev', 'pore'): ([], []),
        },
        num_nodes_dict={'atom': 1, 'pore': 1},
    )
    G.nodes['atom'].data['h'] = torch.zeros(1, n_atom_features)
    G.nodes['pore'].data['h'] = torch.zeros(1, 2)
    G.edges['wall'].data['d']     = torch.zeros(0, 1)
    G.edges['wall_rev'].data['d'] = torch.zeros(0, 1)
    return G


# Dataset wrapper

class PoreGraphDataset(Dataset):
    """Wraps an existing ALIGNN DGL dataset and appends bipartite pore graphs.

    Parameters
    ----------
    base_dataset  : the DGLDataset from get_train_val_loaders
    pore_graph_lookup : dict mapping jid (str) -> DGL heterograph or None
                        (produced by precompute_pore_graphs.py)
    """

    def __init__(self, base_dataset, pore_graph_lookup: dict):
        self.base   = base_dataset
        self.lookup = pore_graph_lookup
        self._empty = _make_empty_pore_graph()

    def __getattr__(self, name):
        # Forward attribute access so DataLoader can reach .ids etc.
        return getattr(self.base, name)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        base_item = self.base[idx]           # (..., target)
        jid       = str(self.base.ids[idx])
        G         = self.lookup.get(jid, None)
        if G is None:
            G = self._empty
        # Insert G just before the target -- same convention as pore_utils.py
        return (*base_item[:-1], G, base_item[-1])


# Collate function

def _collate_with_pore_graph(samples):
    """Batch a list of (g, lg, lat, G_pore, target) tuples.

    DGL heterographs are batched with dgl.batch(), which stacks them into
    a single disconnected heterograph -- the standard approach for GNN batching.
    """
    # Unzip: each sample is (g, lg, lat, G_pore, target)
    gs, lgs, lats, G_pores, targets = zip(*samples)

    batched_g      = dgl.batch(list(gs))
    batched_lg     = dgl.batch(list(lgs))
    batched_lat    = torch.stack(list(lats))
    batched_G_pore = dgl.batch(list(G_pores))   # batches heterographs correctly
    batched_target = torch.tensor(list(targets))

    return batched_g, batched_lg, batched_lat, batched_G_pore, batched_target


# Convenience loader builder

def get_pore_graph_loaders(config, pore_graphs_path: str):
    """Return train/val/test loaders with bipartite pore graphs attached.

    Parameters
    ----------
    config           : TrainingConfig
    pore_graphs_path : path to the .pt file from precompute_pore_graphs.py

    Returns
    -------
    train_loader, val_loader, test_loader, prepare_batch
    """
    from alignn.data import get_train_val_loaders

    line_graph = config.compute_line_graph > 0

    # build plain loaders (same as pore_utils.py)
    train_loader, val_loader, test_loader, prepare_batch = get_train_val_loaders(
        dataset=config.dataset,
        target=config.target,
        n_train=config.n_train,
        n_val=config.n_val,
        n_test=config.n_test,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        batch_size=config.batch_size,
        atom_features=config.atom_features,
        neighbor_strategy=config.neighbor_strategy,
        standardize=config.atom_features != "cgcnn",
        line_graph=line_graph,
        id_tag=config.id_tag,
        pin_memory=config.pin_memory,
        workers=config.num_workers,
        save_dataloader=config.save_dataloader,
        use_canonize=config.use_canonize,
        filename=config.filename,
        cutoff=config.cutoff,
        max_neighbors=config.max_neighbors,
        output_features=config.model.output_features,
        classification_threshold=config.classification_threshold,
        target_multiplication_factor=config.target_multiplication_factor,
        standard_scalar_and_pca=config.standard_scalar_and_pca,
        keep_data_order=config.keep_data_order,
        output_dir=config.output_dir,
        use_lmdb=config.use_lmdb,
        dtype=config.dtype,
    )

    # load precomputed pore graphs
    print(f'Loading precomputed pore graphs from {pore_graphs_path}')
    pore_graph_lookup = torch.load(pore_graphs_path, weights_only=False)
    print(f'  Loaded {len(pore_graph_lookup)} entries')

    # wrap datasets
    train_dataset = PoreGraphDataset(train_loader.dataset, pore_graph_lookup)
    val_dataset   = PoreGraphDataset(val_loader.dataset,   pore_graph_lookup)
    test_dataset  = PoreGraphDataset(test_loader.dataset,  pore_graph_lookup)

    # rebuild loaders with custom collate_fn
    def _rebuild(original_loader, wrapped_dataset):
        return GraphDataLoader(
            wrapped_dataset,
            batch_size=original_loader.batch_size,
            shuffle=isinstance(
                original_loader.sampler,
                torch.utils.data.RandomSampler,
            ),
            drop_last=original_loader.drop_last,
            num_workers=original_loader.num_workers,
            pin_memory=original_loader.pin_memory,
            collate_fn=_collate_with_pore_graph,
        )

    new_train = _rebuild(train_loader, train_dataset)
    new_val   = _rebuild(val_loader,   val_dataset)
    new_test  = _rebuild(test_loader,  test_dataset)

    print(
        f'Pore-graph splits: '
        f'train={len(train_dataset)}  '
        f'val={len(val_dataset)}  '
        f'test={len(test_dataset)}'
    )

    return new_train, new_val, new_test, prepare_batch


# Sanity check

def sanity_check(train_loader):
    """Print batch shapes to verify pore graphs are present."""
    batch = next(iter(train_loader))
    g, lg, lat, G_pore, target = batch
    print('Batch contents:')
    print(f'  g       : DGLGraph   nodes={g.num_nodes()}  edges={g.num_edges()}')
    print(f'  lg      : DGLGraph   nodes={lg.num_nodes()}  edges={lg.num_edges()}')
    print(f'  lat     : {lat.shape}')
    print(f'  G_pore  : DGL HeteroGraph')
    print(f'            atom nodes = {G_pore.num_nodes("atom")}  '
          f'pore nodes = {G_pore.num_nodes("pore")}  '
          f'edges = {G_pore.num_edges("wall")}')
    print(f'            atom features: {G_pore.nodes["atom"].data["h"].shape}')
    print(f'            pore features: {G_pore.nodes["pore"].data["h"].shape}')
    print(f'  target  : {target.shape}')
    print('\nSanity check passed.')
