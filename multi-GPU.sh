#!/bin/bash
#SBATCH -A atm112
#SBATCH -J FCNet
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err


source ~/miniconda_frontier/etc/profile.d/conda.sh
conda deactivate #leave the base conda environemnt. delete this line if base environment not activated

ulimit -n 65536
source source_env.sh

export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=info

export MIOPEN_DISABLE_CACHE=1
export NCCL_PROTO=Simple
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH


export OMP_NUM_THREADS=7
export PYTHONPATH=$PWD:$PYTHONPATH


config='afno_backbone'
run_num='0'

time srun -n $((SLURM_JOB_NUM_NODES*8)) \
       	python train_ddp.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num

