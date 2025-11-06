#!/usr/bin/bash
#SBATCH -J bge-m3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=32G
#SBATCH -w aurora-g8
#SBATCH -p batch_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

pwd

# modified from https://github.com/FlagOpen/FlagEmbedding/blob/master/Tutorials/7_Fine-tuning/7.1.2_Fine-tune.ipynb
torchrun --nproc_per_node 1 \
--master_port 22334 \
-m bgem3ctrl \
    --model_name_or_path dragonkue/bge-m3-ko \
    --cache_dir /data2/local_datasets/encoder/cache/3ep/ctrl/model \
    --cache_path /data2/local_datasets/encoder/cache/3ep/ctrl/data \
    --train_data /data2/local_datasets/encoder/dataset/incidents_ft_minedHN_demo.jsonl \
    --output_dir /data2/local_datasets/encoder/output/3ep/ctrl \
    --save_steps 1000 \
    --logging_steps 200 \
    --num_train_epochs 3 \
    --dataloader_drop_last True \
    --unified_finetuning True \
    --learning_rate 1e-5 \
    --fp16 \
    --pad_to_multiple_of 8 \
    --per_device_train_batch_size 2 \
    --normalize_embeddings True \
    --temperature 0.02 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --train_group_size 8 \
    --negatives_cross_device \
    --seed 42 \
    --data_seed 42 \
    --load_best_model_at_end True \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --metric_for_best_model loss \
    --greater_is_better False \
    --eval_data /data2/local_datasets/encoder/dataset/incidents_val_minedHN_demo.jsonl \