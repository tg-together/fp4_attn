#!/bin/bash
set -e  # exit on error

Initialize conda
eval "$(conda shell.bash hook)"

# Create env
conda create -n fp4 python=3.12 -y

# Activate env
conda activate fp4

# Install packages
pip install lm_eval==0.4.9.1 matplotlib glog


BASHRC="$HOME/.bashrc"

# Remove old CUDA lines (if any)
sed -i '/export CUDA_HOME=\/usr\/local\/cuda/d' "$BASHRC"
sed -i '/export PATH=\$CUDA_HOME\/bin:\$PATH/d' "$BASHRC"
sed -i '/export LD_LIBRARY_PATH=\$CUDA_HOME\/lib64:\$LD_LIBRARY_PATH/d' "$BASHRC"

# Add new CUDA 12.9 lines
cat << 'EOF' >> "$BASHRC"

# >>> CUDA 12.9 default setup >>>
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# <<< CUDA 12.9 default setup <<<
EOF

source "$BASHRC"
# Install local package if directory exists
if [ -d "fast_fp4" ]; then
  cd fast_fp4
  pip install -e .
else
  echo "Directory fast_fp4 not found. Skipping local install."
fi

rm -rf /home/tanmaey/miniconda3/envs/fp4/lib/python3.12/site-packages/lm_eval/tasks/gpqa
rm -rf /home/tanmaey/miniconda3/envs/fp4/lib/python3.12/site-packages/lm_eval/tasks/gsm8k
rm -rf /home/tanmaey/miniconda3/envs/fp4/lib/python3.12/site-packages/lm_eval/tasks/gsm8k_platinum
cd tasks
cp aime/ gpqa/ gsm8k/ gsm8k_platinum/ /home/tanmaey/miniconda3/envs/fp4/lib/python3.12/site-packages/lm_eval/tasks/