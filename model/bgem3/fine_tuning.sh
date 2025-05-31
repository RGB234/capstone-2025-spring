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
    --cache_dir /data2/local_datasets/encoder/cache/bgem3/model \
"

data_args="\
    --train_data /data2/local_datasets/encoder/data/relevant_incidents_train_minedHN.jsonl \
    --cache_path /data2/local_datasets/encoder/cache/bgem3/data \
    --train_group_size 2 \
    --query_max_len 256 \
    --passage_max_len 256 \
    --pad_to_multiple_of 8 \
    
"

training_args="\
    --output_dir /data2/local_datasets/encoder/bgem3/ft \
    --unified_finetuning True \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --dataloader_drop_last True \
    --normalize_embeddings True \
    --temperature 0.02 \
    --negatives_cross_device \
    --logging_steps 10 \
    --save_steps 1000 \
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


