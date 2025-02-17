#!/bin/bash
if [ -f .env ]; then
    export $(xargs < .env)
fi

# shellcheck source=/home/hamis/miniconda3/bin/activate
source ~/miniconda3/bin/activate
conda activate "$CONDA_ENV"

python pysrc/run_gan_tests.py \
    --datafile "$DATA_DIR/$DATA_PREFIX-matrices.txt" 