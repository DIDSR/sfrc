conda activate horovod

# ----------------------------------------------------# 
#     SRGAN training parameters
# ----------------------------------------------------# 
SCALE=4
NEPOCH=20
NGPU=4
MAIN_PATH=""
TRAIN_SR_PY="${MAIN_PATH}/main_gan_hvd.py"
TRAIN_FNAME="${MAIN_PATH}/train_sm_6p_p96_x${SCALE}/"
VAL_FNAME="${MAIN_PATH}/p96_val_uni_ind_airT_x${SCALE}.h5"
DES_TYPE="p96_uni_augTrTaTdT"

time horovodrun -np 4 -H localhost:4 python $TRAIN_SR_PY \
--batch-size 32 --batches-per-allreduce 1 --cuda \
--nepochs $NEPOCH --base-lr 5e-5 --training-fname $TRAIN_FNAME \
--validating-fname $VAL_FNAME --scale $SCALE \
--descriptor-type $DES_TYPE --val-chk-prsc 'positive-float' \
--val-batch-size 32 --shuffle_patches --save_log_ckpts

# ----------------------------------------------------# 
#     SR-WGAN training parameters
# ----------------------------------------------------# 
SCALE=4
NEPOCH=20
NGPU=4
MAIN_PATH=""
TRAIN_SR_PY="${MAIN_PATH}/main_wgan_hvd.py"
TRAIN_FNAME="${MAIN_PATH}/train_sm_6p_p128_x${SCALE}"
VAL_FNAME="${MAIN_PATH}/p128_val_uni_ind_airT_x${SCALE}.h5"
DES_TYPE="p128_uni_ind_augTrTaTdsT_x${SCALE}"

time horovodrun -np $NGPUS -H localhost:$NGPUS python ${TRAIN_SR_PY} \
--batch-size 64 --batches-per-allreduce 1 --cuda \
--nepochs $NEPOCH --base-lr 1e-4 --training-fname ${TRAIN_FNAME} \
--validating-fname ${VAL_FNAME} --scale $SCALE \
--descriptor-type $DES_TYPE --val-chk-prsc 'positive-float' \
--val-batch-size 32 --shuffle_patches \
--gan-name 'srwgan'  --gen-lambda 1e3 --save_log_ckpts