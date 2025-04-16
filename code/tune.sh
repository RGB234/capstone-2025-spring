#!/usr/bin/bash
#SBATCH -J bge-m3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -w aurora-g3
#SBATCH -p batch_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

pwd

torchrun --nproc_per_node 1 \
-m FlagEmbedding.finetune.embedder.encoder_only.base \
    --model_name_or_path BAAI/bge-m3 \
    --cache_dir /data2/local_datasets/bge-m3/cache/model \
    --cache_path /data2/local_datasets/bge-m3/cache/data \
    --train_data /data2/local_datasets/bge-m3/finetune_data_minedHN.jsonl \
    --output_dir /data2/local_datasets/bge-m3/finetuned_bge-m3 \
    --learning_rate 1e-5 \
    --fp16 \
    --pad_to_multiple_of 8 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --dataloader_drop_last True \
    --normalize_embeddings True \
    --temperature 0.02 \
    --query_max_len 64 \
    --passage_max_len 256 \
    --train_group_size 2 \
    --negatives_cross_device \
    --logging_steps 10 \
    --save_steps 1000 \
    --query_instruction_for_retrieval ""

destroy_process_group()

exit 0