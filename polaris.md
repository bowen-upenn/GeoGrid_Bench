To run on polaris, you can use the following example command:
```bash
qsub -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 -q debug -A ARAIA
module use /soft/modulefiles
module load conda ; conda activate base
cd /lus/eagle/projects/ARAIA/xinyu/multimodal_climate_benchmark

conda activate geospatial

curl https://sh.rustup.rs -sSf | sh
source $HOME/.cargo/env

bash scripts/run_inference_llama_3p1_8b.sh text 0 -1

bash scripts/run_inference_llama_3p2_11b_vision.sh image 0 -1
```

