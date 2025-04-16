#!/bin/bash

# Activate the virtual environment (if needed)
# source myenv/bin/activate

# Get modality from the first argument, default to "all"
MODALITY="${1:-all}"

# Get start and end indices from the second and third arguments, default to 0
START_IDX="${2:-0}"
END_IDX="${3:--1}"   # If -1 is passed, it will be ignored and the entire dataset will be used

# Arguments for the Python script
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
QUESTION_PATH="data/benchmark/qa_data.csv"
RESULT_PATH="result/eval_results_${MODEL_NAME}_${MODALITY}.jsonl"

# Base command
CMD=(
  python "inference.py"
  --model "$MODEL_NAME"
  --question_path "$QUESTION_PATH"
  --modality "$MODALITY"
  --result_path "$RESULT_PATH"
  --start_idx "$START_IDX"
  --end_idx "$END_IDX"
)

# Add --clean only if START_IDX is 0
if [ "$START_IDX" -eq 0 ]; then
  CMD+=(--clean)
fi

# Run the command
"${CMD[@]}"


# Example Usage
# bash scripts/run_inference_claude_3p5_haiku.sh all 0 100