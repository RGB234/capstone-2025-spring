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
--master_port 22334 \
-m finetune \
    --model_name_or_path dragonkue/bge-m3-ko \
    --cache_dir /data2/local_datasets/encoder/cache/bgem3_custom/model \
    --cache_path /data2/local_datasets/encoder/cache/bgem3_custom/data \
    --train_data /data2/local_datasets/encoder/data/relevant_incidents_train_minedHN.jsonl \
    --eval_data /data2/local_datasets/encoder/data/relevant_incidents_test_minedHN.jsonl \
    --load_best_model_at_end True \
    --include_for_metrics loss \
    --eval_strategy steps \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 100 \
    --num_train_epochs 5 \
    --dataloader_drop_last True \
    --output_dir /data2/local_datasets/encoder/bgem3_custom/ft \
    --unified_finetuning True \
    --learning_rate 1e-5 \
    --fp16 \
    --pad_to_multiple_of 8 \
    --per_device_train_batch_size 1 \
    --normalize_embeddings True \
    --temperature 0.02 \
    --query_max_len 256 \
    --passage_max_len 256 \
    --train_group_size 2 \
    --negatives_cross_device \
    
    