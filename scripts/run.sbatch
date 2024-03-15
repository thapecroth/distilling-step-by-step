#!/bin/bash -l
#SBATCH --job-name=distill
#SBATCH --partition=gpu-a40
#SBATCH --account=balazinska
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --output="/gscratch/balazinska/enhaoz/distilling-step-by-step/scripts/outputs/slurm-%j-%x.out"

cat $0
echo "Date" $(date)
echo "--------------------"

dataset=$1
r=$2
lora_alpha=$3
lr=$4
batch_size=$5

echo "dataset: $dataset"
echo "r: $r"
echo "lora_alpha: $lora_alpha"
echo "lr: $lr"
echo "batch_size: $batch_size"

cd /gscratch/balazinska/enhaoz/distilling-step-by-step
bash /gscratch/balazinska/enhaoz/distilling-step-by-step/scripts/apptainer-script.sh \
    python run.py --from_pretrained google/t5-v1_1-base --dataset $dataset --model_type task_prefix --label_type gt --llm palm --alpha 0.5 --batch_size $batch_size --qlora_train --r $r --lora_alpha $lora_alpha --lr $lr
