##############################################################
# @author: pkc 
#
# banded_plots_4r_sfrc_ct_tuning.py
# ............................................................
# includes codes to output different banded plots shown in the
# sfrc paper's supplemental document. These plots correspond
# to frequency wise image components of ct images used to 
# tune sFRC. 
# comment the line mpi.use('Agg') in the file src/plot_func.py
# to view subplots as displayed output

import sys
sys.path.append('../')
import numpy as np
import numpy.fft as fft
from src import frc_utils
from src import io_func
from src import frc_utils
from src import plot_func as pf 
import os
from src import utils

def get3_comp_of_img(img, band1_ep, band2_ep, band3_ep):
    h, w   = img.shape
    img_p1 = np.zeros([h,w]).astype('complex')
    img_p2 = np.zeros([h,w]).astype('complex')
    img_p3 = np.zeros([h,w]).astype('complex')
    img_fft = fft.fftshift(fft.fft2(img))

    indices=frc_utils.ring_indices(img_fft)
    for i in range(0, band1_ep):
        img_p1[indices[i]]=img_fft[indices[i]]
    for i in range(band1_ep, band2_ep):
        img_p2[indices[i]]=img_fft[indices[i]]
    for i in range(band2_ep, band3_ep):
        img_p3[indices[i]]=img_fft[indices[i]]

    img_p1 = np.real(fft.ifft2(fft.ifftshift(img_p1)))
    img_p2 = np.real(fft.ifft2(fft.ifftshift(img_p2)))
    img_p3 = np.real(fft.ifft2(fft.ifftshift(img_p3)))
    img_stack = np.stack((img.reshape(h,w), img_p1.reshape(h, w), img_p2.reshape(h, w), img_p3.reshape(h, w)), axis=0)
    return(img_p1, img_p2, img_p3, img_stack)

def get3_fft_bands_of_img(img, band1_ep, band2_ep, band3_ep):
    h, w   = img.shape
    img_p1 = np.zeros([h,w])
    img_p2 = np.zeros([h,w])
    img_p3 = np.zeros([h,w])
    img_fft = np.log(np.abs(fft.fftshift(fft.fft2(img))))

    indices=frc_utils.ring_indices(img_fft)
    for i in range(0, band1_ep):
        img_p1[indices[i]]=img_fft[indices[i]]
    for i in range(band1_ep, band2_ep):
        img_p2[indices[i]]=img_fft[indices[i]]
    for i in range(band2_ep, band3_ep):
        img_p3[indices[i]]=img_fft[indices[i]]
    img_stack = np.stack((img_fft.reshape(h,w), img_p1.reshape(h, w), img_p2.reshape(h, w), img_p3.reshape(h, w)), axis=0)
    return(img_p1, img_p2, img_p3, img_stack)

def get5_comp_of_img(img, band1_ep, band2_ep, band3_ep, band4_ep, band5_ep):
    h, w   = img.shape
    img_p1 = np.zeros([h,w]).astype('complex')
    img_p2 = np.zeros([h,w]).astype('complex')
    img_p3 = np.zeros([h,w]).astype('complex')
    img_p4 = np.zeros([h,w]).astype('complex')
    img_p5 = np.zeros([h,w]).astype('complex')
    img_fft = fft.fftshift(fft.fft2(img))

    indices=frc_utils.ring_indices(img_fft)
    for i in range(0, band1_ep):
        img_p1[indices[i]]=img_fft[indices[i]]
    for i in range(band1_ep, band2_ep):
        img_p2[indices[i]]=img_fft[indices[i]]
    for i in range(band2_ep, band3_ep):
        img_p3[indices[i]]=img_fft[indices[i]]
    for i in range(band3_ep, band4_ep):
        img_p4[indices[i]]=img_fft[indices[i]]
    for i in range(band4_ep, band5_ep):
        img_p5[indices[i]]=img_fft[indices[i]]

    img_p1 = np.real(fft.ifft2(fft.ifftshift(img_p1)))
    img_p2 = np.real(fft.ifft2(fft.ifftshift(img_p2)))
    img_p3 = np.real(fft.ifft2(fft.ifftshift(img_p3)))
    img_p4 = np.real(fft.ifft2(fft.ifftshift(img_p4)))
    img_p5 = np.real(fft.ifft2(fft.ifftshift(img_p5)))
    img_stack = np.stack((img.reshape(h,w), img_p1.reshape(h, w), img_p2.reshape(h, w), img_p3.reshape(h, w), img_p4.reshape(h, w), img_p5.reshape(h, w)), axis=0)
    return(img_p1, img_p2, img_p3, img_p4, img_p5, img_stack)

def get5_fft_bands_of_img(img, band1_ep, band2_ep, band3_ep, band4_ep, band5_ep):
    h, w   = img.shape
    img_p1 = np.zeros([h,w])
    img_p2 = np.zeros([h,w])
    img_p3 = np.zeros([h,w])
    img_p4 = np.zeros([h,w])
    img_p5 = np.zeros([h,w])
    img_fft = np.log(np.abs(fft.fftshift(fft.fft2(img))))

    indices=frc_utils.ring_indices(img_fft)
    for i in range(0, band1_ep):
        img_p1[indices[i]]=img_fft[indices[i]]
    for i in range(band1_ep, band2_ep):
        img_p2[indices[i]]=img_fft[indices[i]]
    for i in range(band2_ep, band3_ep):
        img_p3[indices[i]]=img_fft[indices[i]]
    for i in range(band3_ep, band4_ep):
        img_p4[indices[i]]=img_fft[indices[i]]
    for i in range(band4_ep, band5_ep):
        img_p5[indices[i]]=img_fft[indices[i]]
    img_stack = np.stack((img_fft.reshape(h,w), img_p1.reshape(h, w), img_p2.reshape(h, w), img_p3.reshape(h, w), img_p4.reshape(h, w), img_p5.reshape(h, w) ), axis=0)
    return(img_p1, img_p2, img_p3, img_p4, img_p5, img_stack)

# ----------------------------------------
# Displaying and saving options
# -----------------------------------------
plot_fig   = True  # display image
save_fig   = True  # Save banded plots
Ncomp      = 5     # no. of binned fft components. Other option are not available

# ---------------------------------------------------------------------------
# the five tuning images used for the CT super resolution problem------------
tuning_img_str_arr = ['000039', '000069', '000149', '000189', '000217']
# ---------------------------------------------------------------------------
img_str            = tuning_img_str_arr[0]
gt_path            = './data/test_sh_L067/ua_ll_3mm_HR_tune_x4/'+ img_str +'.raw'
cnn_path           = './results/sh_L067/ua_ll_smSRGAN_tune_in_x4/checkpoint-generator-20/test_sh_L067_cnn/' + img_str +'.raw'

out_path_gt_b1     = './data/test_sh_L067/tuning_images/b1/gt/'
out_path_gt_b2     = './data/test_sh_L067/tuning_images/b2/gt/'
out_path_gt_b3     = './data/test_sh_L067/tuning_images/b3/gt/'
out_path_gt_b4     = './data/test_sh_L067/tuning_images/b4/gt/'
out_path_gt_b5     = './data/test_sh_L067/tuning_images/b4/gt/'

out_path_cnn_b1    = './data/test_sh_L067/tuning_images/b1/cnn/'
out_path_cnn_b2    = './data/test_sh_L067/tuning_images/b2/cnn/'
out_path_cnn_b3    = './data/test_sh_L067/tuning_images/b3/cnn/'
out_path_cnn_b4    = './data/test_sh_L067/tuning_images/b4/cnn/'
out_path_cnn_b5    = './data/test_sh_L067/tuning_images/b4/cnn/'

out_path_diff_full= './data/test_sh_L067/tuning_images/full_img/diff/'
out_path_gt_full  = './data/test_sh_L067/tuning_images/full_img/gt/'
out_path_cnn_full = './data/test_sh_L067/tuning_images/full_img/cnn/'

out_path_diff_b1  = './data/test_sh_L067/tuning_images/b1/diff/'
out_path_diff_b2  = './data/test_sh_L067/tuning_images/b2/diff/'
out_path_diff_b3  = './data/test_sh_L067/tuning_images/b3/diff/'
out_path_diff_b4  = './data/test_sh_L067/tuning_images/b4/diff/'
out_path_diff_b5  = './data/test_sh_L067/tuning_images/b5/diff/'


if not os.path.isdir(out_path_gt_b1): os.makedirs(out_path_gt_b1, exist_ok=True)
if not os.path.isdir(out_path_gt_b2): os.makedirs(out_path_gt_b2, exist_ok=True)
if not os.path.isdir(out_path_gt_b3): os.makedirs(out_path_gt_b3, exist_ok=True)
if not os.path.isdir(out_path_gt_b4): os.makedirs(out_path_gt_b4, exist_ok=True)
if not os.path.isdir(out_path_gt_b5): os.makedirs(out_path_gt_b5, exist_ok=True)

if not os.path.isdir(out_path_cnn_b1): os.makedirs(out_path_cnn_b1, exist_ok=True)
if not os.path.isdir(out_path_cnn_b2): os.makedirs(out_path_cnn_b2, exist_ok=True)
if not os.path.isdir(out_path_cnn_b3): os.makedirs(out_path_cnn_b3, exist_ok=True)
if not os.path.isdir(out_path_cnn_b4): os.makedirs(out_path_cnn_b4, exist_ok=True)
if not os.path.isdir(out_path_cnn_b5): os.makedirs(out_path_cnn_b5, exist_ok=True)

if not os.path.isdir(out_path_diff_b1): os.makedirs(out_path_diff_b1, exist_ok=True)
if not os.path.isdir(out_path_diff_b2): os.makedirs(out_path_diff_b2, exist_ok=True)
if not os.path.isdir(out_path_diff_b3): os.makedirs(out_path_diff_b3, exist_ok=True)
if not os.path.isdir(out_path_diff_b4): os.makedirs(out_path_diff_b4, exist_ok=True)
if not os.path.isdir(out_path_diff_b5): os.makedirs(out_path_diff_b5, exist_ok=True)

if not os.path.isdir(out_path_diff_full): os.makedirs(out_path_diff_full, exist_ok=True)
if not os.path.isdir(out_path_gt_full): os.makedirs(out_path_gt_full, exist_ok=True)
if not os.path.isdir(out_path_cnn_full): os.makedirs(out_path_cnn_full, exist_ok=True)


h, w         = (512, 512)
r            = int(h/2)
gt_img       = io_func.raw_imread(gt_path, (h, w), dtype='uint16')
cnn_img      = io_func.raw_imread(cnn_path, (h, w), dtype='uint16')

gt_img_wind  = pf.img_windowing(gt_img, windowing='fk_ct_soft')
cnn_img_wind = pf.img_windowing(cnn_img, windowing='fk_ct_soft')


fs_c1  = int(0.1*r)  # very low-frequency region
fs_c2  = int(0.25*r) # low-frequency region
fs_c3  = int(0.5*r)  # mid-frequency region
fs_c4  = int(0.75*r) # high-frequency region 

cnn_c1, cnn_c2, cnn_c3, cnn_c4, cnn_c5, cnn_c_stack = get5_comp_of_img(cnn_img, fs_c1, fs_c2, fs_c3, fs_c4, r) # missing-wedge-img/srgan-patch
gt_c1,  gt_c2,  gt_c3, gt_c4, gt_c5, gt_c_stack     = get5_comp_of_img(gt_img, fs_c1, fs_c2, fs_c3, fs_c4, r)
cnn_fft_c1, cnn_fft_c2, cnn_fft_c3, cnn_fft_c4, cnn_fft_c5, cnn_fft_c_stack = get5_fft_bands_of_img(gt_img, fs_c1, fs_c2, fs_c3, fs_c4, r)
diff_c_stack = (gt_c_stack - cnn_c_stack)


# ----------------------------------------------------------------------------------------------------------------------
# Display plot plots of different image components for different images
# along the rows: subplots with image components corresponding the 5 frequency-based regions
# along the rows: filter, normal-resolution-based reference, SRGAN and difference image between (reference and SRGAN)
# ----------------------------------------------------------------------------------------------------------------------
if plot_fig:
    #pf.multi2dplots(1, 6, cnn_c_stack, axis=0, passed_fig_att={'colorbar': False})
    #pf.multi2dplots(1, 6, gt_c_stack, axis=0,  passed_fig_att={'colorbar': False})
    #pf.multi2dplots(1, 6, cnn_fft_c_stack, axis=0, passed_fig_att={'colorbar': False})
    #pf.plot2dlayers(gt_img)

    band_stacks = np.stack((cnn_fft_c_stack.reshape(6, h, w), gt_c_stack.reshape(6, h, w), cnn_c_stack.reshape(6, h, w), diff_c_stack.reshape(6, h, w)),axis=0)
    print('Shape of the subplot', (band_stacks.reshape(24, h, w)).shape)
    #out_banded_img_name = ('./banded_plots/full_fig/Ncomp_5_uint8_all_aw.pdf')
    pf.multi2dplots(4, 6, band_stacks.reshape(24, h, w), axis=0, passed_fig_att={'colorbar': False, 'figsize': [8, 6]})#, 'out_path': out_banded_img_name })#, 'out_path': out_path + 'all_bands.png'})

# ------------------------------------------------------------------
# normalize and save the image components as uint8 
# ------------------------------------------------------------------
if save_fig:        
    # SRGAN parts ---------------------------------------------------------------------------
    cnn_c1 = utils.normalize_data_ab(0, 255, cnn_c1).astype('uint8')
    cnn_c2 = utils.normalize_data_ab(0, 255, cnn_c2).astype('uint8')
    cnn_c3 = utils.normalize_data_ab(0, 255, cnn_c3).astype('uint8')
    cnn_c4 = utils.normalize_data_ab(0, 255, cnn_c4).astype('uint8')
    cnn_c5 = utils.normalize_data_ab(0, 255, cnn_c5).astype('uint8')
    
    #Normal resolution (reference) parts ------------------------------------------------------
    gt_c1 = utils.normalize_data_ab(0, 255, gt_c1).astype('uint8')
    gt_c2 = utils.normalize_data_ab(0, 255, gt_c2).astype('uint8')
    gt_c3 = utils.normalize_data_ab(0, 255, gt_c3).astype('uint8')
    gt_c4 = utils.normalize_data_ab(0, 255, gt_c4).astype('uint8')
    gt_c5 = utils.normalize_data_ab(0, 255, gt_c5).astype('uint8')

    # difference image (between reference and SRGAN) as image components -----------------------
    diff_c1 = utils.normalize_data_ab(0, 255, diff_c_stack[1]).astype('uint8')
    diff_c2 = utils.normalize_data_ab(0, 255, np.log(np.abs(diff_c_stack[2]))).astype('uint8')
    diff_c3 = utils.normalize_data_ab(0, 255, np.log(np.abs(diff_c_stack[3]))).astype('uint8')
    diff_c4 = utils.normalize_data_ab(0, 255, diff_c_stack[4]).astype('uint8')
    diff_c5 = utils.normalize_data_ab(0, 255, diff_c_stack[5]).astype('uint8')
    
    # Fourier domain-based imaging components of the SRGAN -------------------------------------
    cnn_fft_c1 = utils.normalize_data_ab(0, 255, cnn_fft_c1).astype('uint8')
    cnn_fft_c2 = utils.normalize_data_ab(0, 255, cnn_fft_c2).astype('uint8')
    cnn_fft_c3 = utils.normalize_data_ab(0, 255, cnn_fft_c3).astype('uint8')
    cnn_fft_c4 = utils.normalize_data_ab(0, 255, cnn_fft_c4).astype('uint8')
    cnn_fft_c5 = utils.normalize_data_ab(0, 255, cnn_fft_c5).astype('uint8')
    cnn_fft    = utils.normalize_data_ab(0, 255, cnn_fft_c_stack[0]).astype('uint8')

    # actual images without any fourier frequency-based decompositions -------------------------
    diff_full = utils.normalize_data_ab(0, 255, diff_c_stack[0]).astype('uint8')
    gt_full   = utils.normalize_data_ab(0, 255, gt_img_wind).astype('uint8')
    cnn_full  = utils.normalize_data_ab(0, 255, cnn_img_wind).astype('uint8')
    
    io_func.imsave(cnn_c1, path=out_path_cnn_b1 + img_str +'.png', svtype='original')
    io_func.imsave(cnn_c2, path=out_path_cnn_b2 + img_str +'.png', svtype='original')
    io_func.imsave(cnn_c3, path=out_path_cnn_b3 + img_str +'.png', svtype='original')
    io_func.imsave(cnn_c4, path=out_path_cnn_b4 + img_str +'.png', svtype='original')
    io_func.imsave(cnn_c5, path=out_path_cnn_b5 + img_str +'.png', svtype='original')

    
    io_func.imsave(diff_c1, path=out_path_diff_b1 + img_str +'.png', svtype='original')
    io_func.imsave(diff_c2, path=out_path_diff_b2 + img_str +'.png', svtype='original')
    io_func.imsave(diff_c3, path=out_path_diff_b3 + img_str +'.png', svtype='original')
    io_func.imsave(diff_c4, path=out_path_diff_b4 + img_str +'.png', svtype='original')
    io_func.imsave(diff_c5, path=out_path_diff_b5 + img_str +'.png', svtype='original')

    io_func.imsave(gt_c1, path=out_path_gt_b1+ img_str +'.png', svtype='original')
    io_func.imsave(gt_c2, path=out_path_gt_b2 + img_str +'.png', svtype='original')
    io_func.imsave(gt_c3, path=out_path_gt_b3 + img_str +'.png', svtype='original')
    io_func.imsave(gt_c4, path=out_path_gt_b4 + img_str +'.png', svtype='original')
    io_func.imsave(gt_c5, path=out_path_gt_b5 + img_str +'.png', svtype='original')

    io_func.imsave(diff_full, path=out_path_diff_full + img_str +'.png', svtype='original')
    io_func.imsave(gt_full, path=out_path_gt_full + img_str +'.png', svtype='original')
    io_func.imsave(cnn_full, path=out_path_cnn_full + img_str +'.png', svtype='original')

