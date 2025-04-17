#!/bin/bash -l
#PBS -N run
#PBS -l select=2:ncpus=64:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A ARAIA

# Set the number of ranks per node (4 GPUs per node)
export NRANKS_PER_NODE=4
export NTOTRANKS=8  # Total ranks (2 nodes * 4 GPUs)
export NDEPTH=8     # CPU cores per rank (64 cores / 8 ranks)

module use /soft/modulefiles
module load conda
conda activate base
conda activate geospatial

cd /lus/eagle/projects/ARAIA/xinyu/multimodal_climate_benchmark/

# Create GPU affinity script
cat > set_affinity_gpu_polaris.sh << 'EOF'
#!/bin/bash -l
num_gpus=4
gpu=$((${num_gpus} - 1 - ${PMI_LOCAL_RANK} % ${num_gpus}))
export CUDA_VISIBLE_DEVICES=$gpu
echo "RANK= ${PMI_RANK} LOCAL_RANK= ${PMI_LOCAL_RANK} gpu= ${gpu}"
exec "$@"
EOF
chmod +x set_affinity_gpu_polaris.sh

# Create bash script for running commands
cat > run_inference_task.sh << 'EOF'
#!/bin/bash

# Get rank from MPI environment
RANK=$PMI_RANK
SIZE=$PMI_SIZE

# Define all tasks
declare -a TASKS=(
  "bash ./scripts/run_inference_llama_3p2_11b_vision.sh text 0 -1"
  "bash ./scripts/run_inference_llama_3p2_11b_vision.sh code 0 -1"
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
EOF
chmod +x run_inference_task.sh

# Launch MPI job
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth \
./set_affinity_gpu_polaris.sh ./run_inference_task.sh