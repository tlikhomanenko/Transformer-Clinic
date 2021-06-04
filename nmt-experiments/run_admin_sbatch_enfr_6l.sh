#!/bin/bash

#SBATCH --cpus-per-task=10
#SBATCH --partition=devlab
#SBATCH --gres=gpu:volta:8
#SBATCH --ntasks-per-node=8
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

GPUID=0
TOKEN_NUMBER=8000 
UPDATE_FREQUENCE=10
cd /checkpoint/antares/experiments/fl_new_master/augposemb/Transforemr-Clinic/nmt-experiments

CUDA_VISIBLE_DEVICES=$GPUID fairseq-train ../data-bin/wmt14_en_fr_joined_dict \
	--arch transformer_wmt_en_de --share-all-embeddings --optimizer radam \
	--adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
	--warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.0007 --min-lr 1e-09 \
	--dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 --weight-decay 0.0 \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--max-tokens $TOKEN_NUMBER --update-freq $UPDATE_FREQUENCE \
	--fp16 --fp16-scale-window 256 --threshold-loss-scale 0.03125 \
	--seed 1111 --restore-file x.pt --max-epoch 50 --save-dir wmt14enfr/wmt-admin-6l \
	--user-dir ../radam_fairseq --init-type adaptive-profiling \
  --profiling_path wmt14enfr/wmt-admin-6l/profile.ratio.init \
	--log-format simple --log-interval 100 | tee ./wmt14enfr/log/loss-admin-6l.log

fairseq-train ../data-bin/wmt14_en_fr_joined_dict \
	--arch transformer_wmt_en_de --share-all-embeddings --optimizer radam \
	--adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
	--warmup-init-lr 1e-07 --warmup-updates 8000 --lr 0.0007 --min-lr 1e-09 \
	--dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 --weight-decay 0.0 \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--max-tokens $TOKEN_NUMBER --update-freq $UPDATE_FREQUENCE \
	--fp16 --fp16-scale-window 256 --threshold-loss-scale 0.03125 \
	--seed 1111 --restore-file x.pt --max-epoch 50 --save-dir wmt14enfr/wmt-admin-6l \
	--user-dir ../radam_fairseq --init-type adaptive \
  --profiling_path wmt14enfr/wmt-admin-6l/profile.ratio.init \
	--log-format simple --log-interval 100 | tee ./wmt14enfr/log/loss-admin-6l.log

bash eval_wmt_en-fr.sh ./wmt14enfr/wmt-admin-6l/ $GPUID

