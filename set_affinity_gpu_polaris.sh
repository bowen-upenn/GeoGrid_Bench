#!/bin/bash -l
num_gpus=4 # Total number of GPUs available on the node
gpus_per_rank=4 # Number of GPUs you want each rank to see

# Calculate the starting GPU for the current local rank
# This assumes that ranks are grouped and assigned GPUs contiguously.
start_gpu=$(( (PMI_LOCAL_RANK * gpus_per_rank) % num_gpus ))

# Construct the CUDA_VISIBLE_DEVICES string
CUDA_VISIBLE_DEVICES=""
for i in $(seq 0 $((gpus_per_rank - 1))); do
    current_gpu=$(( (start_gpu + i) % num_gpus ))
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        CUDA_VISIBLE_DEVICES="$current_gpu"
    else
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES,$current_gpu"
    fi
done

export CUDA_VISIBLE_DEVICES
echo "RANK= ${PMI_RANK} LOCAL_RANK= ${PMI_LOCAL_RANK} CUDA_VISIBLE_DEVICES= ${CUDA_VISIBLE_DEVICES}"
exec "$@"