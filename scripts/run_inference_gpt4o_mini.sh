#!/bin/bash

# Limit virtual memory to 32GB (in KB)
ulimit -v $((32 * 1024 * 1024))

# Activate the virtual environment (if needed)
# source myenv/bin/activate

# Arguments for the Python script
MODEL_NAME="gpt-4o-mini"
QUESTION_PATH="output/qa_data.csv"
MODALITY="all"
RESULT_PATH="result/eval_results_${MODEL_NAME}_${MODALITY}.jsonl"

# Run the Python script with the specified arguments
python "inference.py" --model "$MODEL_NAME" --question_path "$QUESTION_PATH" --modality "$MODALITY" --result_path "$RESULT_PATH" --clean