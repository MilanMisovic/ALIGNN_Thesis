"""pore_utils.py, pore-feature support for ALIGNN on hMOF.
 
This module handles everything between the raw hMOF JARVIS dataset and
the training loop:
 
  1. extract_pore_features(): pull the 3-4 Zeo++ scalars from hMOF rows
                                  and return a normalised numpy array.
  2. PoreAugmentedDataset: wraps the existing ALIGNN DGL dataset and
                                  attaches per-structure pore tensors so the
                                  DataLoader yields them automatically.
  3. get_pore_loaders(): convenience wrapper around
                                  alignn.data.get_train_val_loaders that
                                  returns PoreAugmented loaders ready for
                                  use with the modified train.py / alignn.py.
PORE COLUMNS USED
Column name            Physical meaning

void_fraction          Fraction of unit cell that is void (0—1)
pld                    Pore limiting diameter (Å), bottleneck
lcd                    Largest cavity diameter (Å), widest sphere
surface_area_m2cm3     Volumetric surface area (m²/cm³)
 
All five are already precomputed in the hMOF JARVIS dataset, no need
to run Zeo++ yourself.
"""
 
from __future__ import annotations
 
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List
 

# The five pore columns we extract from every hMOF entry
 
PORE_COLS: List[str] = [
    "void_fraction",        # dimensionless, 0–1
    "pld",                  # A
    "lcd",                  # A
    #"surface_area_m2cm3",   # m2/cm3
    # NOTE: 'density' does NOT exist as a column in the hMOF JARVIS dataset.
    # It was removed after discovering all rows were dropped by dropna() because
    # row.get('density') returns NaN for every entry.
    # Density can be derived as surface_area_m2cm3 / surface_area_m2g if needed,
    # but is not included here since it is redundant given the other two ASA columns.
]
 
# Approximate statistics computed on the full hMOF dataset (137 953 MOFs).
# We hard-code them so we can normalise without scanning the whole dataset
# at every run.  If you retrain on a different dataset replace these.
#
# Format: {col: (mean, std)}
_PORE_STATS = {
    "void_fraction":      (0.6206,  0.227),
    "pld":                (7.4548,   4.4906),
    "lcd":                (8.9843,  4.4549),
    #"surface_area_m2cm3": (1652.3213, 761.5194),
}
 
 
def extract_pore_features(
    row: dict,
    cols: List[str] = PORE_COLS,
    normalise: bool = True,
) -> np.ndarray:
    """Extract and optionally z-score normalise pore scalars from one hMOF row.
 
    row        : dict-like entry from the JARVIS hMOF dataset
    cols       : list of column names to extract (default: PORE_COLS)
    normalise  : if True, apply (x - mean) / std using _PORE_STATS
 
    Returns
    np.ndarray of shape (len(cols),)   float32
    """
    vals = []
    for col in cols:
        v = float(row.get(col, 0.0))
        if np.isnan(v) or np.isinf(v):
            v = 0.0          # safe fallback for missing data
        if normalise and col in _PORE_STATS:
            mean, std = _PORE_STATS[col]
            v = (v - mean) / (std + 1e-8)
        vals.append(v)
    return np.array(vals, dtype=np.float32)
 
 
#
# Dataset wrapper that attaches per-structure pore tensors
#
 
class PoreAugmentedDataset(Dataset):
    """Wraps an existing ALIGNN DGL dataset and appends pore features.
 
    The wrapped dataset's __getitem__ returns:
        (g, [lg], [lat], pore_tensor, target)
 
    instead of the original:
        (g, [lg], [lat], target)
 
    The pore_tensor is inserted just before the target so that
    _unpack_batch() in train.py can find it at dats[-2] and the
    target at dats[-1] regardless of line_graph mode.
 
    Parameters
    ----------
    base_dataset  : the DGLDataset object from get_train_val_loaders
    pore_matrix   : np.ndarray of shape (N, n_pore) — one row per
                    structure in the same order as base_dataset.
    """
 
    def __init__(self, base_dataset: Dataset, pore_matrix: np.ndarray):
        self.base = base_dataset
        # Pre-convert to tensors for speed
        self.pore = torch.tensor(pore_matrix, dtype=torch.float32)
 
        if len(self.base) != len(self.pore):
            raise ValueError(
                f"base_dataset has {len(self.base)} entries but "
                f"pore_matrix has {len(self.pore)} rows. They must match."
            )
 
    # Forward attribute access so the DataLoader can still reach
    # .ids, .close(), and any other attributes on the base dataset.
    def __getattr__(self, name):
        return getattr(self.base, name)
 
    def __len__(self):
        return len(self.base)
 
    def __getitem__(self, idx):
        base_item = self.base[idx]   # tuple: (..., target)
        pore_vec  = self.pore[idx]   # Tensor (n_pore,)
 
        # Insert pore_vec just before the last element (target).
        # base_item[-1] is always the target scalar/tensor.
        return (*base_item[:-1], pore_vec, base_item[-1])
 
 
#
# Convenience loader wrapper
#
 
def get_pore_loaders(
    config,
    pore_cols: List[str] = PORE_COLS,
    normalise_pore: bool = True,
):
    """Return train/val/test loaders augmented with pore features.
 
    Wraps alignn.data.get_train_val_loaders() but rebuilds each DataLoader
    around a PoreAugmentedDataset instead of trying to swap .dataset on an
    already-initialised GraphDataLoader (which PyTorch forbids).

    config          : TrainingConfig with dataset, target, split ratios, etc.
    pore_cols       : which Zeo++ scalars to include (default: all 5)
    normalise_pore  : whether to z-score the pore scalars (recommended)
 
    Returns
    train_loader, val_loader, test_loader, prepare_batch
        — same interface as get_train_val_loaders, ready to pass to train_dgl
          via the train_val_test_loaders argument.
    """
    from alignn.data import get_train_val_loaders
    from jarvis.db.figshare import data as jdata
    from dgl.data.utils import Subset
    from dgl.dataloading import GraphDataLoader
 
    line_graph = config.compute_line_graph > 0
 
    # build the plain (no-pore) loaders
    # We use these to get the datasets and the prepare_batch function.
    # The loaders themselves are discarded and rebuilt below.
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
 
    # load the full hMOF table to get pore columns
    print(f"Loading hMOF pore descriptors for columns: {pore_cols}")
    hmof_data = jdata(config.dataset)   # list of dicts
    id_tag = config.id_tag


    # PERMUTATION TEST: swap this block in to verify no leakage
    # Replace real features with random noise.
    # Expected: MAE should collapse back to ~188 (no-feature baseline).
    # If MAE stays low (~100), there is a data pipeline issue.
    # TO ENABLE:  change False -> True
    # TO DISABLE: change True  -> False  (normal training)
    _PERMUTATION_TEST = False
    if _PERMUTATION_TEST:
        print("  *** PERMUTATION TEST ACTIVE — pore features are RANDOM NOISE ***")
        pore_lookup = {
            str(row[id_tag]): np.random.randn(len(pore_cols)).astype(np.float32)
            for row in hmof_data
        }
    else:
        pore_lookup = {
            str(row[id_tag]): extract_pore_features(row, pore_cols, normalise_pore)
            for row in hmof_data
        }
    # End permutation test block

    print(f"  Loaded {len(pore_lookup)} entries.")
 

    # build pore matrices aligned to each split's .ids list
    def _pore_matrix(loader):
        ids = loader.dataset.ids
        return np.stack([
            pore_lookup.get(str(i), np.zeros(len(pore_cols), dtype=np.float32))
            for i in ids
        ])  # shape: (N_split, n_pore)
 
    train_pore = _pore_matrix(train_loader)
    val_pore   = _pore_matrix(val_loader)
    test_pore  = _pore_matrix(test_loader)
 
    # wrap the datasets with pore features
    # We wrap the underlying dataset objects (not the loaders) and then
    # build brand-new GraphDataLoaders around them.
    # This avoids the PyTorch/DGL restriction that forbids setting
    # .dataset on an already-initialised DataLoader.
    train_dataset = PoreAugmentedDataset(train_loader.dataset, train_pore)
    val_dataset   = PoreAugmentedDataset(val_loader.dataset,   val_pore)
    test_dataset  = PoreAugmentedDataset(test_loader.dataset,  test_pore)
 
    # rebuild the GraphDataLoaders with the wrapped datasets
    # Mirror the settings from the original loaders so shuffle / workers /
    # batch size etc. are all preserved.
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
            collate_fn=original_loader.collate_fn
            if hasattr(original_loader, "collate_fn")
            and original_loader.collate_fn is not None
            else None,
        )
 
    new_train_loader = _rebuild(train_loader, train_dataset)
    new_val_loader   = _rebuild(val_loader,   val_dataset)
    new_test_loader  = _rebuild(test_loader,  test_dataset)
 
    print(
        f"Pore-augmented splits: "
        f"train={len(train_pore)}, "
        f"val={len(val_pore)}, "
        f"test={len(test_pore)}"
    )
 
    return new_train_loader, new_val_loader, new_test_loader, prepare_batch
 
 
#
# Quick sanity-check you can run in a notebook cell
#
 
def sanity_check(train_loader, n_pore: int = 3):
    """Print the shape of the first batch to verify pore tensors are present.
 
    Call this right after get_pore_loaders() to confirm everything is wired
    up before you start a long training run.
 
    Expected output (line_graph=True, batch_size=4, pore_features=5):
        Batch element shapes:
          [0] g   : DGLGraph  num_nodes=...
          [1] lg  : DGLGraph  num_nodes=...
          [2] lat : torch.Size([4, 3, 3])
          [3] pore: torch.Size([4, 5])     <-- new element
          [4] tgt : torch.Size([4])
    """
    batch = next(iter(train_loader))
    print("Batch element shapes:")
    labels = ["g", "lg", "lat", "pore", "tgt"]
    for i, item in enumerate(batch):
        label = labels[i] if i < len(labels) else f"[{i}]"
        if hasattr(item, "num_nodes"):
            print(f"  [{i}] {label:4s}: DGLGraph  num_nodes={item.num_nodes()}")
        else:
            print(f"  [{i}] {label:4s}: {item.shape}")
 
    # Verify pore tensor is in the right place (second to last element)
    pore_idx = len(batch) - 2
    pore_tensor = batch[pore_idx]
    assert pore_tensor.shape[-1] == n_pore, (
        f"Expected pore tensor width {n_pore}, got {pore_tensor.shape[-1]}. "
        "Check that config.model.pore_features matches len(pore_cols)."
    )
    print(f"\nSanity check passed — pore tensor shape: {pore_tensor.shape}")
 
