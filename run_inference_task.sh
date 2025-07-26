#!/bin/bash

# Get rank from MPI environment
RANK=$PMI_RANK
SIZE=$PMI_SIZE

# Define all tasks
declare -a TASKS=(
  "bash ./scripts/run_inference_internvl_2p5_8b.sh image 1 1600"
  "bash ./scripts/run_inference_internvl_2p5_8b.sh image 1600 2000"
)

# Calculate which tasks this rank should handle
for i in "${!TASKS[@]}"; do
  if [ $((i % SIZE)) -eq $RANK ]; then
    echo "RANK $RANK executing task $i: ${TASKS[$i]}"
    eval "${TASKS[$i]}"
    echo "RANK $RANK completed task $i"
  fi
done
