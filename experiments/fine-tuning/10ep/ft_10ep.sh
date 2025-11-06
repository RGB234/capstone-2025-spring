#!/usr/bin/bash
#SBATCH -J bge-m3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=32G
#SBATCH -w aurora-g6
#SBATCH -p batch_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

export PYTHONPATH=/data/fehs0611/repos/custom_encoder/custom_bgem3/

# sh 실행
echo "Running bgem3_10ep.sh..."
bash ./scripts/bgem3_10ep.sh
if [ $? -ne 0 ]; then
    echo "bgem3_10ep.sh failed. Exiting."
    exit 1
fi

# sh 실행
echo "Running ctrl_10ep.sh..."
bash ./scripts/ctrl_10ep.sh
if [ $? -ne 0 ]; then
    echo "ctrl_10ep.sh failed. Exiting."
    exit 1
fi

# sh 실행
echo "Running saq_10ep.sh..."
bash ./scripts/saq_10ep.sh
if [ $? -ne 0 ]; then
    echo "saq_10ep.sh failed. Exiting."
    exit 1
fi

echo "All scripts executed successfully!"
