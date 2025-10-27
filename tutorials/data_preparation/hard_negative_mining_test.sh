#!/usr/bin/bash
#SBATCH -J bge-m3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -w aurora-g8
#SBATCH -p batch_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

pwd

python hard_negative_mining.py \
--embedder_name_or_path BAAI/bge-m3 \
--input_file /data2/local_datasets/encoder/data/incidents_test.jsonl \
--output_file /data2/local_datasets/encoder/data/incidents_test_minedHN.jsonl \
--range_for_sampling 2-300 \
--negative_number 8 \
--use_gpu_for_searching

exit 0