# pkc created
import os
import numpy as np
import scipy
import imageio

# Function for converting float32 image array to uint8 array in the range [0,255]
def convert_to_uint(img):
    img = 255 * (img-img.min())/(img.max()-img.min())
    return img.astype(np.uint8)

for i in range(5):
    recon     = np.load('gt_ood/gt_'+str(i)+'.npy')
    mask      = np.load('seg_mask_ood/sm_'+str(i)+'.npy')
    recon_mk  = recon*mask     
    recon_mk_int = convert_to_uint(recon_mk)
    imageio.imwrite('masked__gt_ood/gt_'+str(i)+'.png', recon_mk_int)