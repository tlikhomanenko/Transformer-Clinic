#!/bin/bash

#SBATCH --cpus-per-task=10
#SBATCH --partition=devlab,learnlab,learnfair
#SBATCH --gres=gpu:volta:4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --time=72:00:00
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
GPUS=0,1,2,3
GPUID=1
TOKEN_NUMBER=8192
UPDATE_FREQUENCE=1
lnum=$1
gl=$2
cd /checkpoint/antares/experiments/fl_new_master/augposemb/Transforemr-Clinic/nmt-experiments

CUDA_VISIBLE_DEVICES=$GPUID fairseq-train \
  ../data-bin/wmt14_en_de_joined_dict/ -s en -t de \
  --arch transformer_wmt_en_de --share-all-embeddings \
  --optimizer radam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --max-update 500000 \
  --warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.001 --min-lr 1e-09  \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --weight-decay 0.0 --attention-dropout 0.1 --relu-dropout 0.1 \
  --max-tokens $TOKEN_NUMBER --update-freq $UPDATE_FREQUENCE \
  --save-dir wmt14ende/wmt-admin-cape-${lnum}l-gl-${gl} --restore-file x.pt --seed 1111 \
  --user-dir ../radam_fairseq --log-format simple --log-interval 500 \
  --init-type adaptive-profiling --profiling_path wmt14ende/wmt-admin-cape-${lnum}l-gl-${gl}/profile.ratio.init \
  --fp16 --fp16-scale-window 256 \
  --encoder-layers $lnum --decoder-layers $lnum --augmented-embedding --global-shift=${gl} \
  --threshold-loss-scale 0.03125 | tee ./wmt14ende/log/loss_admin-cape-${lnum}l-gl-${gl}.log

fairseq-train \
  ../data-bin/wmt14_en_de_joined_dict/ -s en -t de \
  --arch transformer_wmt_en_de --share-all-embeddings \
  --optimizer radam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --max-update 500000 \
  --warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.001 --min-lr 1e-09  \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --weight-decay 0.0 --attention-dropout 0.1 --relu-dropout 0.1 \
  --max-tokens $TOKEN_NUMBER --update-freq $UPDATE_FREQUENCE \
  --save-dir wmt14ende/wmt-admin-cape-${lnum}l-gl-${gl} --restore-file x.pt --seed 1111 \
  --user-dir ../radam_fairseq --log-format simple --log-interval 500 \
  --init-type adaptive --profiling_path wmt14ende/wmt-admin-cape-${lnum}l-gl-${gl}/profile.ratio.init \
  --fp16 --fp16-scale-window 256 \
  --encoder-layers $lnum --decoder-layers $lnum --augmented-embedding --global-shift=${gl} \
  --threshold-loss-scale 0.03125 | tee ./wmt14ende/log/loss_admin-cape-${lnum}l-gl-${gl}.log


bash eval_wmt_en-de.sh wmt14ende/wmt-admin-cape-${lnum}l-gl-${gl} $GPUID 




