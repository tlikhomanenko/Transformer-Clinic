#!/bin/bash

#SBATCH --cpus-per-task=10
#SBATCH --partition=prioritylab
#SBATCH --gres=gpu:volta:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=110GB
#SBATCH --signal=B:USR1@200
#SBATCH --comment=Neurips
#SBATCH --open-mode=append
#SBATCH --job-name=mt_admin
#SBATCH --output=/checkpoint/antares/experiments/fl_new_master/augposemb/Transforemr-Clinic/nmt-experiments/sbatch_log/%j.out
#SBATCH --error=/checkpoint/antares/experiments/fl_new_master/augposemb/Transforemr-Clinic/nmt-experiments/sbatch_log/%j.err

# 1. Load modules
module purge
module load cuda/11.0
module load cudnn/v8.0.3.33-cuda.11.0
module load NCCL/2.8.3-1-cuda.11.0
module load intel/mkl/2020.3.279
module load anaconda3/5.0.1
source deactivate
source activate /private/home/antares/.conda/envs/pytorch

# 3. Your job
NCCL_SOCKET_IFNAME=^docker0,lo
cd /checkpoint/antares/experiments/fl_new_master/augposemb/Transforemr-Clinic/nmt-experiments

for (( i=25; i<=50; i+=1 ))
do
CUDA_VISIBLE_DEVICES=0 fairseq-generate ../data-bin/wmt14_en_de_joined_dict \
                    --path $1/checkpoint$i.pt \
                    --batch-size 128 --beam 4 --lenpen 0.6 --remove-bpe \
                    --user-dir ../radam_fairseq --quiet --gen-subset valid
CUDA_VISIBLE_DEVICES=0 fairseq-generate ../data-bin/wmt14_en_de_joined_dict \
                    --path $1/checkpoint$i.pt \
                    --batch-size 128 --beam 4 --lenpen 0.6 --remove-bpe \
                    --user-dir ../radam_fairseq --quiet --gen-subset test
done
