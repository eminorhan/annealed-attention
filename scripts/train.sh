#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=1:00:00
#SBATCH --job-name=train_annealed_attention
#SBATCH --output=train_annealed_attention_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

srun python -u ../train.py \
	--model 'mae_vit_huge_patch14' \
	--resume '' \
	--accum_iter 1 \
	--epochs 100 \
	--batch_size_per_gpu 1024 \
	--input_size 224 \
	--mask_ratio 0.8 \
	--lr 0.0001 \
	--min_lr 0.0001 \
	--weight_decay 0.0 \
	--num_workers 16 \
	--output_dir /scratch/eo41/annealed-attention/outputs/models_pretrained \
	--data_path /scratch/work/public/imagenet/train \
	--save_prefix vith14_${SLURM_ARRAY_TASK_ID} \
	--compile

echo "Done"