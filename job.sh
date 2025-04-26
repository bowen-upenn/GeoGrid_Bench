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

curl https://sh.rustup.rs -sSf | sh
source $HOME/.cargo/env

unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/cudnn/cudnn-cuda12-linux-x64-v9.4.0.58/lib:/soft/compilers/cudatoolkit/cuda-12.4.1/extras/CUPTI/lib64:/soft/compilers/cudatoolkit/cuda-12.4.1/lib64:/soft/libraries/trt/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0/lib:/soft/libraries/nccl/nccl_2.21.5-1+cuda12.4_x86_64/lib:/soft/perftools/darshan/darshan-3.4.4/lib:/opt/cray/pe/papi/7.0.1.2/lib64:/opt/cray/libfabric/1.15.2.0/lib64
unset LD_PRELOAD

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

chmod +x run_inference_task.sh

# Launch MPI job
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth \
./set_affinity_gpu_polaris.sh ./run_inference_task.sh