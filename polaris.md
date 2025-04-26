To run on polaris, you can use the following example command:
```bash
qsub -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 -q debug -A ARAIA
module use /soft/modulefiles
module load conda ; conda activate base
cd /lus/eagle/projects/ARAIA/xinyu/multimodal_climate_benchmark

conda activate geospatial



bash scripts/run_inference_llama_3p1_8b.sh text 0 -1

bash scripts/run_inference_llama_3p2_11b_vision.sh image 0 -1


curl https://sh.rustup.rs -sSf | sh
source $HOME/.cargo/env

unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/cudnn/cudnn-cuda12-linux-x64-v9.4.0.58/lib:/soft/compilers/cudatoolkit/cuda-12.4.1/extras/CUPTI/lib64:/soft/compilers/cudatoolkit/cuda-12.4.1/lib64:/soft/libraries/trt/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0/lib:/soft/libraries/nccl/nccl_2.21.5-1+cuda12.4_x86_64/lib:/soft/perftools/darshan/darshan-3.4.4/lib:/opt/cray/pe/papi/7.0.1.2/lib64:/opt/cray/libfabric/1.15.2.0/lib64
unset LD_PRELOAD

bash ./scripts/run_inference_qwen_2p5_vl_7b.sh image 0 100
```

