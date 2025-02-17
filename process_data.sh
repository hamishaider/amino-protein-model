#!/bin/bash
if [ -f .env ]; then
    export $(xargs < .env)
fi

# shellcheck source=/home/hamis/miniconda3/bin/activate
source ~/miniconda3/bin/activate
conda activate "$CONDA_ENV"
python pysrc/process_data_generate_intermediate.py "$DATA_DIR/$DATA_PREFIX-raw.txt"
python pysrc/process_data_generate_matrices.py "$DATA_DIR/$DATA_PREFIX-intermediate.txt"