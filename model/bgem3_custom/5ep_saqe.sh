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

# modified from https://github.com/FlagOpen/FlagEmbedding/blob/master/Tutorials/7_Fine-tuning/7.1.2_Fine-tune.ipynb
torchrun --nproc_per_node 1 \
--master_port 22334 \
-m bgem3saqe \
    --model_name_or_path dragonkue/bge-m3-ko \
    --cache_dir /data2/local_datasets/encoder/cache/bgem3_custom/5ep_saqe/model \
    --cache_path /data2/local_datasets/encoder/cache/bgem3_custom/5ep_saqe/data \
    --train_data /data2/local_datasets/encoder/data/relevant_incidents_train_minedHN.jsonl \
    --save_steps 500 \
    --logging_steps 100 \
    --num_train_epochs 5 \
    --dataloader_drop_last True \
    --output_dir /data2/local_datasets/encoder/bgem3_custom/5ep_saqe \
    --unified_finetuning True \
    --learning_rate 1e-5 \
    --fp16 \
    --pad_to_multiple_of 8 \
    --per_device_train_batch_size 1 \
    --normalize_embeddings True \
    --temperature 0.02 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --train_group_size 2 \
    --negatives_cross_device \
    --seed 42 \
    --data_seed 42 \
    
    