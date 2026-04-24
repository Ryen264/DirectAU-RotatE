#!/usr/bin/env bash

set -e

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=codes
DATA_PATH=data
SAVE_PATH=models
CONFIG_PATH=configs

print_usage() {
    echo "Usage:"
    echo "  bash run.sh train MODEL DATASET GPU_DEVICE SAVE_ID BATCH_SIZE NEGATIVE_SAMPLE_SIZE HIDDEN_DIM GAMMA ALPHA LEARNING_RATE MAX_STEPS TEST_BATCH_SIZE [EXTRA_FLAGS]"
    echo "  bash run.sh valid MODEL DATASET GPU_DEVICE SAVE_ID"
    echo "  bash run.sh test MODEL DATASET GPU_DEVICE SAVE_ID"
    echo "  bash run.sh config_FILE.yaml"
}

yaml_get() {
    local key="$1"
    local file="$2"
    sed -n "s/^[[:space:]]*${key}:[[:space:]]*//p" "$file" | sed 's/[[:space:]]*#.*$//' | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//' | sed "s/^['\"]//; s/['\"]$//" | head -n 1
}

resolve_config_file() {
    local config_input="$1"
    if [[ -f "$config_input" ]]; then
        echo "$config_input"
    elif [[ -f "$CONFIG_PATH/$config_input" ]]; then
        echo "$CONFIG_PATH/$config_input"
    else
        echo ""
    fi
}

run_command() {
    local MODE="$1"
    local MODEL="$2"
    local DATASET="$3"
    local GPU_DEVICE="$4"
    local SAVE_ID="$5"

    local FULL_DATA_PATH="$DATA_PATH/$DATASET"
    local SAVE="$SAVE_PATH/${MODEL}_${DATASET}_${SAVE_ID}"

    if [[ "$MODE" == "train" ]]; then
        local BATCH_SIZE="$6"
        local NEGATIVE_SAMPLE_SIZE="$7"
        local HIDDEN_DIM="$8"
        local GAMMA="$9"
        local ALPHA="${10}"
        local LEARNING_RATE="${11}"
        local MAX_STEPS="${12}"
        local TEST_BATCH_SIZE="${13}"
        local EXTRA_FLAGS=("${@:14}")

        echo "Start Training......"

        CUDA_VISIBLE_DEVICES="$GPU_DEVICE" python -u "$CODE_PATH/run.py" --do_train \
            --cuda \
            --do_valid \
            --do_test \
            --data_path "$FULL_DATA_PATH" \
            --model "$MODEL" \
            -n "$NEGATIVE_SAMPLE_SIZE" -b "$BATCH_SIZE" -d "$HIDDEN_DIM" \
            -g "$GAMMA" -a "$ALPHA" -adv \
            -lr "$LEARNING_RATE" --max_steps "$MAX_STEPS" \
            -save "$SAVE" --test_batch_size "$TEST_BATCH_SIZE" \
            "${EXTRA_FLAGS[@]}"
    elif [[ "$MODE" == "valid" ]]; then
        echo "Start Evaluation on Valid Data Set......"
        CUDA_VISIBLE_DEVICES="$GPU_DEVICE" python -u "$CODE_PATH/run.py" --do_valid --cuda -init "$SAVE"
    elif [[ "$MODE" == "test" ]]; then
        echo "Start Evaluation on Test Data Set......"
        CUDA_VISIBLE_DEVICES="$GPU_DEVICE" python -u "$CODE_PATH/run.py" --do_test --cuda -init "$SAVE"
    else
        echo "Unknown MODE $MODE"
        print_usage
        exit 1
    fi
}

if [[ $# -lt 1 ]]; then
    print_usage
    exit 1
fi

if [[ "$1" == *.yaml || "$1" == *.yml ]]; then
    CONFIG_FILE=$(resolve_config_file "$1")
    if [[ -z "$CONFIG_FILE" ]]; then
        echo "Config file not found: $1"
        echo "Expected either '$1' or '$CONFIG_PATH/$1'"
        exit 1
    fi

    MODE=$(yaml_get "mode" "$CONFIG_FILE")
    MODEL=$(yaml_get "model" "$CONFIG_FILE")
    DATASET=$(yaml_get "dataset" "$CONFIG_FILE")
    GPU_DEVICE=$(yaml_get "gpu_device" "$CONFIG_FILE")
    SAVE_ID=$(yaml_get "save_id" "$CONFIG_FILE")

    if [[ -z "$MODE" || -z "$MODEL" || -z "$DATASET" || -z "$GPU_DEVICE" || -z "$SAVE_ID" ]]; then
        echo "Config file missing one of required fields: mode, model, dataset, gpu_device, save_id"
        exit 1
    fi

    if [[ "$MODE" == "train" ]]; then
        BATCH_SIZE=$(yaml_get "batch_size" "$CONFIG_FILE")
        NEGATIVE_SAMPLE_SIZE=$(yaml_get "negative_sample_size" "$CONFIG_FILE")
        HIDDEN_DIM=$(yaml_get "hidden_dim" "$CONFIG_FILE")
        GAMMA=$(yaml_get "gamma" "$CONFIG_FILE")
        ALPHA=$(yaml_get "alpha" "$CONFIG_FILE")
        LEARNING_RATE=$(yaml_get "learning_rate" "$CONFIG_FILE")
        MAX_STEPS=$(yaml_get "max_steps" "$CONFIG_FILE")
        TEST_BATCH_SIZE=$(yaml_get "test_batch_size" "$CONFIG_FILE")
        EXTRA_FLAGS=$(yaml_get "extra_flags" "$CONFIG_FILE")
        EXTRA_FLAGS_ARR=()
        if [[ -n "$EXTRA_FLAGS" ]]; then
            # Split YAML extra_flags string into arguments, e.g. "-de --countries".
            read -r -a EXTRA_FLAGS_ARR <<< "$EXTRA_FLAGS"
        fi

        if [[ -z "$BATCH_SIZE" || -z "$NEGATIVE_SAMPLE_SIZE" || -z "$HIDDEN_DIM" || -z "$GAMMA" || -z "$ALPHA" || -z "$LEARNING_RATE" || -z "$MAX_STEPS" || -z "$TEST_BATCH_SIZE" ]]; then
            echo "Train config missing required fields."
            exit 1
        fi

        run_command "$MODE" "$MODEL" "$DATASET" "$GPU_DEVICE" "$SAVE_ID" \
            "$BATCH_SIZE" "$NEGATIVE_SAMPLE_SIZE" "$HIDDEN_DIM" "$GAMMA" "$ALPHA" "$LEARNING_RATE" "$MAX_STEPS" "$TEST_BATCH_SIZE" "${EXTRA_FLAGS_ARR[@]}"
    else
        run_command "$MODE" "$MODEL" "$DATASET" "$GPU_DEVICE" "$SAVE_ID"
    fi

    exit 0
fi

#The first four parameters must be provided
MODE=$1
MODEL=$2
DATASET=$3
GPU_DEVICE=$4
SAVE_ID=$5

if [[ -z "$MODE" || -z "$MODEL" || -z "$DATASET" || -z "$GPU_DEVICE" || -z "$SAVE_ID" ]]; then
    print_usage
    exit 1
fi

if [[ "$MODE" == "train" ]]; then
    run_command "$@"
else
    run_command "$MODE" "$MODEL" "$DATASET" "$GPU_DEVICE" "$SAVE_ID"
fi