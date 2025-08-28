#!/bin/bash
#PBS -l select=1:ncpus=2:mem=8gb:ngpus=1
#PBS -l walltime=32:00:00
#PBS -N analysis

export RUN_NAME="cifar_test"

if [ -z "$RUN_NAME" ]; then
  echo "Error: You must provide an RUN_NAME as an environment variable."
  exit 1
fi

start_time=$(date +%s)
echo "Start time: $(date)"

rsync -az --update $HOME/Flexible-Neurons-main $TMPDIR

eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda activate base

nvidia-smi

cd Flexible-Neurons-main
python src/train_initialiser.py --config configs/test.json

end_time=$(date +%s)
echo "End time: $(date)"
SECONDS=$((end_time - start_time))
hours=$((SECONDS / 3600))
minutes=$(((SECONDS % 3600) / 60))
seconds=$((SECONDS % 60))
echo "Total time taken: $hours hour(s) $minutes minute(s) $seconds second(s)"

