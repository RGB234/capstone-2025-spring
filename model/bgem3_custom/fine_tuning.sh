#!/usr/bin/bash
#SBATCH -J bge-m3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=32G
#SBATCH -w aurora-g7
#SBATCH -p batch_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

pwd

torchrun --nproc_per_node 1 \
-m finetune \
    --model_name_or_path dragonkue/bge-m3-ko \
    --cache_dir /data2/local_datasets/encoder/cache/bgem3_custom/model \
    --cache_path /data2/local_datasets/encoder/cache/bgem3_custom/data \
    --train_data /data2/local_datasets/encoder/data/relevant_incidents_train_minedHN.jsonl \
    --output_dir /data2/local_datasets/encoder/bgem3_custom/ft \
    --unified_finetuning True \
    --learning_rate 1e-5 \
    --fp16 \
    --pad_to_multiple_of 8 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --dataloader_drop_last True \
    --normalize_embeddings True \
    --temperature 0.02 \
    --query_max_len 256 \
    --passage_max_len 256 \
    --train_group_size 2 \
    --negatives_cross_device \
    --logging_steps 10 \
    --save_steps 1000 \