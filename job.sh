#!/bin/bash -l
#PBS -N run
#PBS -l select=1:ncpus=64:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A ARAIA


export NRANKS_PER_NODE=1 # Changed from 2 to 1
export NTOTRANKS=1       # Changed from 4 to 2 (2 nodes * 1 rank/node)
export NDEPTH=64         # Changed from 32 to 64 (64 cores / 1 rank = 64 cores/rank) - Each rank gets all CPU cores


module use /soft/modulefiles
module load conda
conda activate base
conda activate geospatial

module unload cudnn
module load cudnn/9.4.0

cd /lus/eagle/projects/ARAIA/xinyu/multimodal_climate_benchmark/

chmod +x set_affinity_gpu_polaris.sh

chmod +x run_inference_task.sh

# Launch MPI job
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth \
./set_affinity_gpu_polaris.sh ./run_inference_task.sh