#!/bin/bash
# source /home/prabhat.kc/anaconda3/base_env.sh
# source /home/prabhat.kc/anaconda3/hmri2_env.sh

# removing previously acquired results
rm -r ../recon_data/masked_tune_unet_recons_ood
rm -r ../recon_data/masked_test_unet_recons_ood

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# this only works in the older GPU with arch less than sm80 
# i.e. it only works on V100 and not in A100

# line below takes in k-space data store in the .h5 file in path singlecoil_test/file_ood.h5
# subsamples by factor 3 using subroutine test_data_transofrm in test_unet.py
# applies the checkpoint and saves output as .h5 file in ./experiment/h_map/reconstructions
python -u models/unet/test_unet.py --mode test --challenge singlecoil --gpus 1 --data-path ./ --exp h_map --checkpoint ./experiments/h_map/epoch\=49.ckpt 
python extract_recons_n_apply_mask.py


mv -f masked_test_unet_recons_ood ../recon_data/
mkdir ../recon_data/masked_tune_unet_recons_ood
# recon_0 is used as tuning set for sFRC threshold
mv ../recon_data/masked_test_unet_recons_ood/recon_0.png ../recon_data/masked_tune_unet_recons_ood 

