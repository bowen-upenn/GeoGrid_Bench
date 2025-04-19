#!/bin/bash

# Get rank from MPI environment
RANK=$PMI_RANK
SIZE=$PMI_SIZE

# Define all tasks
declare -a TASKS=(
  "bash ./scripts/run_inference_llama_3p2_11b_vision.sh image 0 -1"
  "bash ./scripts/run_inference_llama_3p2_3b.sh text 0 -1"
  "bash ./scripts/run_inference_llama_3p2_3b.sh code 0 -1"
  "bash ./scripts/run_inference_llama_3p1_8b.sh text 0 -1"
  "bash ./scripts/run_inference_llama_3p1_8b.sh code 0 -1"
)

# Calculate which tasks this rank should handle
for i in "${!TASKS[@]}"; do
  if [ $((i % SIZE)) -eq $RANK ]; then
    echo "RANK $RANK executing task $i: ${TASKS[$i]}"
    eval "${TASKS[$i]}"
    echo "RANK $RANK completed task $i"
  fi
done
