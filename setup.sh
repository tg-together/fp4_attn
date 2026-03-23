#!/bin/bash
set -e  # exit on error

# Initialize conda
# eval "$(conda shell.bash hook)"

# Create env
# conda create -n fp4 python=3.12 -y

# conda create -p /anvil/projects/x-nairr250415/tgupta/conda_envs/fp4 python=3.12 -y

# # Activate env
# conda activate /anvil/projects/x-nairr250415/tgupta/conda_envs/fp4

# Install packages

# python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# python -m pip install transformers==5.1.0 lighteval==0.13.0 matplotlib glog


# Install same pytorch as CUDA first

BASHRC="$HOME/.bashrc"

# Remove old CUDA lines (if any)
sed -i '/export CUDA_HOME=\/usr\/local\/cuda/d' "$BASHRC"
sed -i '/export PATH=\$CUDA_HOME\/bin:\$PATH/d' "$BASHRC"
sed -i '/export LD_LIBRARY_PATH=\$CUDA_HOME\/lib64:\$LD_LIBRARY_PATH/d' "$BASHRC"

# Add new CUDA 12.9 lines
cat << 'EOF' >> "$BASHRC"

# >>> CUDA default setup >>>
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# <<< CUDA default setup <<<
EOF

source "$BASHRC"
# Install local package if directory exists
if [ -d "fast_fp4" ]; then
  cd fast_fp4
  python -m pip install --no-build-isolation -e .
else
  echo "Directory fast_fp4 not found. Skipping local install."
fi