# ALIGNN Extensions for MOF Accessible Surface Area Prediction

This repository contains the code accompanying the master's/bachelor's
thesis *"Crystal Property Prediction with Atomistic Line Graph Neural
Network"* (Milan Misovic, University of Amsterdam). It extends the
ALIGNN architecture (Choudhary & DeCost, 2021) to predict accessible
surface area (ASA) for metal-organic frameworks (MOFs), using the
hMOF dataset from JARVIS.

## Overview

ALIGNN predicts gravimetric ASA (m²/g) accurately but performs
considerably worse on volumetric ASA (m²/cm³). This work identifies
the cause as an architectural blind spot — ALIGNN's fixed-radius graph
cannot represent empty pore space — and proposes two extensions to
address it:

- **Extension I — Pore descriptor augmentation**: concatenates three
  precomputed scalar pore descriptors (void fraction, LCD, PLD) with
  the pooled crystal embedding before the final prediction layer.
- **Extension II — Bipartite atom–pore graph**: constructs an explicit
  graph connecting pore cavities (from a Voronoi decomposition via
  Zeo++) to their surrounding atoms, processed with a dedicated
  message-passing branch.

## Repository structure
alignn/
alignn/

  models/
  
    alignn.py              # ALIGNN model with Extension I & II
    
    config.py                  # Training configuration (pydantic)
    
    train.py                   # Training loop
    
    data.py                    # Standard ALIGNN data loaders
    
    lmdb_dataset.py            # LMDB-backed graph dataset
    
    pore_utils.py              # Extension I: pore descriptor extraction
    
    bipartite_pore_graph.py    # Extension II: pore graph construction
    
    precompute_pore_graphs.py  # Extension II: batch precomputation

    pore_graph_dataset.py      # Extension II: dataset wrapper + collate

Thesis/# Figures, ablation outputs, analysis notebooks


## Setup

```bash
conda create -n mof python=3.10
conda activate mof
pip install -r requirements.txt    # alignn, dgl, jarvis-tools, pyzeo, torch
```

Extension II additionally requires `pyzeo` for the Voronoi
decomposition (Zeo++(http://www.zeoplusplus.org/)).

## Usage
### Extension I (pore descriptors)

```python
from alignn.pore_utils import get_pore_loaders
from alignn.train import train_dgl
from alignn.config import TrainingConfig

config = TrainingConfig(
    dataset='hmof', target='surface_area_m2g', id_tag='id',
    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
    batch_size=16, epochs=50,
)
config.model.pore_features = 3

loaders = get_pore_loaders(config)
train_dgl(config, train_val_test_loaders=list(loaders))
```

### Extension II (pore graph)

```bash
# 1. Precompute pore graphs (one-time, run from alignn/)
python alignn/precompute_pore_graphs.py

# 2. Train
```
```python
from pore_graph_dataset import get_pore_graph_loaders
from alignn.train import train_dgl
from alignn.config import TrainingConfig

config = TrainingConfig(
    dataset='hmof', target='surface_area_m2g', id_tag='id',
    train_ratio=0.01, val_ratio=0.005, test_ratio=0.005,
    batch_size=4, epochs=5, learning_rate=1e-3,
)
config.model.use_pore_graph = True

loaders = get_pore_graph_loaders(config, pore_graphs_path='pore_graphs_3pct.pt')
train_dgl(config, train_val_test_loaders=list(loaders))
```

See the thesis PDF for full methodology, ablations, and discussion.
## Citation

If you use this code, please cite the accompanying thesis:
```bibtex
@thesis{misovic2026alignn,
  author = {Misovic, Milan},
  title  = {Crystal Property Prediction with Atomistic Line Graph Neural Network},
  
  school = {University of Amsterdam},
  year   = {2026}
}
```
This work builds on ALIGNN:

```bibtex
@article{choudhary2021alignn,
  author  = {Choudhary, Kamal and DeCost, Brian},
  title   = {Atomistic Line Graph Neural Network for improved materials property predictions},
  journal = {npj Computational Materials},
  year    = {2021}
}
```

## License

This project is licensed under the MIT License — see the
[LICENSE](LICENSE) file for details. The base ALIGNN code is licensed
separately by its original authors; see their repository for terms.
