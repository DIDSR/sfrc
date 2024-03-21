#$ -P CDRHID0024
#$ -A prabhat.kc
#$ -cwd
#$ -l h_rt=192:00:00
#$ -l h_vmem=200G
#$ -l gpus=2    #no of gpus ordered in the schedular
#$ -S /bin/sh
#$ -j y
#$ -N denoise_ct_hvd_sge
#$ -o sysout_denoise

source /home/prabhat.kc/anaconda3/base_env.sh
source /home/prabhat.kc/anaconda3/horovod_env.sh
conda activate horovod

MAIN_PATH="/projects01/didsr-aiml/prabhat.kc/code/ct-recon/Denoising/DLdenoise"
TRAIN_PY=/projects01/didsr-aiml/prabhat.kc/code/ct-recon/Denoising/DLdenoise/main_hvd.py

NGPU=2
NEPOCH=10
TRAIN_FNAME="${MAIN_PATH}/train_data/p55_no_norm/train"
VAL_FNAME="${MAIN_PATH}/train_data/p55_no_norm/tune"
DES_TYPE='p55_no_norm/augTrTaTdT'

time horovodrun -np $NGPU -H localhost:$NGPU python $TRAIN_PY --batch-size 64 --batches-per-allreduce 1 --cuda \
--nepochs $NEPOCH --base-lr 1e-5 --training-fname ${TRAIN_FNAME} --validating-fname ${VAL_FNAME} \
--descriptor-type $DES_TYPE --val-chk-prsc 'positive-float' --val-batch-size 64 \
--loss-func 'mse' --model-name 'cnn3' --shuffle_patches --save_log_ckpts