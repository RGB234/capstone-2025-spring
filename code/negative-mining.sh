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
--input_file /data2/local_datasets/bge-m3/ft_data/relevant_incidents.jsonl \
--output_file /data2/local_datasets/bge-m3/ft_data/relevant_incidents_minedHN.jsonl \
--range_for_sampling 2-300 \
--negative_number  \
--use_gpu_for_searching 

exit 0