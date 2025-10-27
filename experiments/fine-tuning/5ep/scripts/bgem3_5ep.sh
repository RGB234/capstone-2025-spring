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


model_args="\
    --model_name_or_path dragonkue/bge-m3-ko \
    --cache_dir /data2/local_datasets/encoder/cache/5ep/bgem3/model \
"

data_args="\
    --train_data /data2/local_datasets/encoder/data/incidents_train_minedHN.jsonl \
    --cache_path /data2/local_datasets/encoder/cache/5ep/bgem3/data \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    
"

training_args="\
    --output_dir /data2/local_datasets/encoder/output/5ep/bgem3 \ 
    --num_train_epochs 5 \
    --save_steps 1000 \
    --logging_steps 200 \
    --seed 42 \
    --data_seed 42
    --learning_rate 1e-5 \
    --temperature 0.02 \
    --unified_finetuning True \
    --fp16 \
    --per_device_train_batch_size 2 \
    --dataloader_drop_last True \
    --normalize_embeddings True \
    --negatives_cross_device \
"
num_gpus=1
cmd="torchrun --nproc_per_node $num_gpus \
    -m FlagEmbedding.finetune.embedder.encoder_only.m3 \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd


