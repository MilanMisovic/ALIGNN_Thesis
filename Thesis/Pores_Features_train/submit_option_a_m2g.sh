#!/bin/sh -f
#SBATCH --job-name=alignn_pore_m2g
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.error
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpuRTX-2nd
#SBATCH --gpus-per-node=1

#Temp working directory on the node's fast local storage
export TMPDIR=/tmp
export WORKDIR=${TMPDIR}/${USER}/alignn_m2g/${SLURM_JOBID}
mkdir -p "$WORKDIR"

#Copy project files to node-local storage (avoids slow NFS reads)
echo "Copying project files to $WORKDIR ..."
rsync -ap "$SLURM_SUBMIT_DIR/." "$WORKDIR/"

cd "$WORKDIR"

# Activate conda environment
# Must use explicit source, SLURM starts a minimal shell without conda init
source /FastHome/milanm/miniconda3/etc/profile.d/conda.sh
conda activate mof

echo "============================================"
echo "Job: ALIGNN + Pore Features -- ASA m2/g"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"
echo "WORKDIR: $WORKDIR"
echo "============================================"

# Sanity checks
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import dgl; print('DGL:', dgl.__version__)"
python -c "from alignn.pore_utils import get_pore_loaders; print('pore_utils OK')"
python -c "from jarvis.db.figshare import data; d=data('hmof'); print(f'hMOF: {len(d)} structures')"

# Training
export OUTPUT_DIR="$WORKDIR/output_m2g"
mkdir -p "$OUTPUT_DIR"

python - << 'PYTHON'
import os, sys
sys.path.insert(0, os.environ["WORKDIR"])

import alignn.pore_utils as pu
from alignn.pore_utils import get_pore_loaders, sanity_check
from alignn.train import train_dgl
from alignn.config import TrainingConfig

pu.PORE_COLS = ["void_fraction", "pld", "lcd"]

config = TrainingConfig(
    dataset       = "hmof",
    target        = "surface_area_m2g",
    id_tag        = "id",
    train_ratio   = 0.8,
    val_ratio     = 0.1,
    test_ratio    = 0.1,
    batch_size    = 64,
    epochs        = 50,
    output_dir    = os.environ["OUTPUT_DIR"],
    num_workers   = 4,
    learning_rate = 1e-3,
)
config.model.pore_features = 3

print("Building loaders...")
loaders = get_pore_loaders(config, pore_cols=pu.PORE_COLS)
sanity_check(loaders[0], n_pore=3)

print("Training...")
train_dgl(config, train_val_test_loaders=list(loaders))
print("Done.")
PYTHON

# Copy results back to submit directory
echo "Copying results back..."
RESULTS_DIR="$SLURM_SUBMIT_DIR/results_m2g_${SLURM_JOBID}"
mkdir -p "$RESULTS_DIR"
rsync -ap "$WORKDIR/output_m2g/" "$RESULTS_DIR/"

echo "============================================"
echo "Finished: $(date)"
echo "Results: $RESULTS_DIR"
echo "============================================"
