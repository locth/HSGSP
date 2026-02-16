#!/bin/bash
# run_correct_experiments.sh

echo "Running HSGSP experiment"
dataset=$1
mode=${2:-eval}

if [ "$dataset" = "CIFAR-10" ] && [ "$mode" = "analyze" ]; then
    echo "Running CIFAR-10 pruning-effect analysis..."
    python analyze_pruning_effect.py \
        --task cifar10 \
        --baseline_model "EXPERIMENT/15022026_154958_cifar10_BASE_ALPHA05/models/best_model_243_0.943.h5" \
        --pruned_root "EXPERIMENT/16022026_092916_cifar10_FREQONLY_ALPHA01" \
        --dataset_split val \
        --decision_alpha 1 \
        --max_batches 256
    echo "Analysis completed!"
    exit 0
fi

if [ "$dataset" = "CIFAR-10" ]; then
    echo "Running CIFAR-10 hybrid baseline..."
    python main.py \
        --task cifar10 \
        --model_path "EXPERIMENT/15022026_154958_cifar10_BASE_ALPHA05/models/best_model_243_0.943.h5" \
        --prune \
        --eval \
        --gpu 0
elif [ "$dataset" = "CIFAR-100" ]; then
    echo "Running CIFAR-100 hybrid baseline..."
    python main.py \
        --task cifar100 \
        --model_path "EXPERIMENT/15022026_154958_cifar10_BASE_ALPHA05/models/best_model_243_0.943.h5" \
        --prune \
        --eval \
        --input_norm "$input_norm" \
        --gpu 0
else
    echo "Dataset not available"
fi

echo "Experiments completed!"
