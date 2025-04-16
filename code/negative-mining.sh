#!/usr/bin/bash
#SBATCH -J bge-m3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=20G
#SBATCH -w aurora-g3
#SBATCH -p batch_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out
  
pwd
which python
hostname

python hn_mine.py \
--embedder_name_or_path BAAI/bge-m3 \
--input_file /data2/local_datasets/bge-m3/ft_data/ft_data.jsonl \
--output_file /data2/local_datasets/bge-m3/ft_data/ft_data_minedHN.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--use_gpu_for_searching 

exit 0