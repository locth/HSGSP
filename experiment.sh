#!/bin/bash
# run_correct_experiments.sh

echo "Running HSGSP experiment"
dataset=$1

if [ "$dataset" = "CIFAR-10" ]; then
    echo "Running CIFAR-10 hybrid baseline..."
    python main.py \
        --task cifar10 \
        --model_path "EXPERIMENT/11022026_171013_cifar10/models/best_model_187_0.944.h5" \
        --pruned_model_path "EXPERIMENT/11022026_171013_cifar10/models/cifar10_hybrid_pruned.h5" \
        --eval \
        --gpu 0
elif [ "$dataset" = "CIFAR-100" ]; then
    echo "Running CIFAR-100 hybrid baseline..."
    python main.py \
        --task cifar100 \
        --model_path "EXPERIMENT/15112025_111805_cifar100_base/models/final_model.h5" \
        --prune \
        --eval \
        --gpu 0
else
    echo "Dataset not available"
fi

echo "Experiments completed!"
