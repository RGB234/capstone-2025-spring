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

model_args="\
    --model_name_or_path dragonkue/bge-m3-ko \
    --cache_dir /data2/local_datasets/encoder/cache/3ep/bgem3/model
"

data_args="\
    --train_data /data2/local_datasets/encoder/dataset/incidents_ft_minedHN_demo.jsonl \
    --cache_path /data2/local_datasets/encoder/cache/3ep/bgem3/data \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --eval_data /data2/local_datasets/encoder/dataset/incidents_val_minedHN_demo.jsonl
"

training_args="\
    --output_dir /data2/local_datasets/encoder/output/3ep/bgem3 \
    --num_train_epochs 3 \
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
    --load_best_model_at_end True \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --metric_for_best_model loss \
    --greater_is_better False
"
num_gpus=1
cmd="torchrun --nproc_per_node $num_gpus \
    -m bgem3 \
    $model_args \
    $data_args \
    $training_args \
"
eval $cmd


