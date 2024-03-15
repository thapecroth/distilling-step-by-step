#!/bin/bash

declare -a DatasetArray=("esnli" "cqa" "svamp")

for dataset in "${DatasetArray[@]}"; do
    for r in 8 16 64; do
        for lora_alpha in 32; do
            for lr in 5e-4 5e-5; do
                for batch_size in 64; do
                    sbatch run.sbatch $dataset $r $lora_alpha $lr $batch_size
                done
            done
        done
    done
done

for r in 8 16 64; do
    for lora_alpha in 32; do
        for lr in 5e-4 5e-5; do
            for batch_size in 32; do
                sbatch run.sbatch "anli1" $r $lora_alpha $lr $batch_size
            done
        done
    done
done