##############################################################
# @author: pkc 
#
# banded_plots.py
# ............
# includes codes to output different banded plots shown in the
# sfrc paper
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

def get5_comp_of_img(img, band1_ep, band2_ep, band3_ep, band4_ep, band5_ep, win_arr=6*[None]):
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

    img_w   = pf.img_windowing(img, windowing=win_arr[0])

    img_p1w = pf.img_windowing(img_p1, windowing=win_arr[1])
    img_p2w = pf.img_windowing(img_p2, windowing=win_arr[2])
    img_p3w = pf.img_windowing(img_p3, windowing=win_arr[3])
    img_p4w = pf.img_windowing(img_p4, windowing=win_arr[4])
    img_p5w = pf.img_windowing(img_p5, windowing=win_arr[5])

    img_stack = np.stack((img_w.reshape(h,w), img_p1w.reshape(h, w), img_p2w.reshape(h, w), img_p3w.reshape(h, w), img_p4w.reshape(h, w), img_p5w.reshape(h, w)), axis=0)
    return(img_p1w, img_p2w, img_p3w, img_p4w, img_p5w, img_stack)

def get5_fft_bands_of_img(img, band1_ep, band2_ep, band3_ep, band4_ep, band5_ep, binary_out=False):
    h, w   = img.shape
    img_p1 = np.zeros([h,w])
    img_p2 = np.zeros([h,w])
    img_p3 = np.zeros([h,w])
    img_p4 = np.zeros([h,w])
    img_p5 = np.zeros([h,w])
    img_fft = np.log(np.abs(fft.fftshift(fft.fft2(img))))

    indices=frc_utils.ring_indices(img_fft)

    if binary_out:
        for i in range(0, band1_ep):
            img_p1[indices[i]]=1.0
        for i in range(band1_ep, band2_ep):
            img_p2[indices[i]]=1.0
        for i in range(band2_ep, band3_ep):
            img_p3[indices[i]]=1.0
        for i in range(band3_ep, band4_ep):
            img_p4[indices[i]]=1.0
        for i in range(band4_ep, band5_ep):
            img_p5[indices[i]]=1.0
    else:
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

# --------------------------------------
# Displaying and saving options
# -----------------------------------------
plot_fig     = True   # Display plots 
save_fig     = False   # save plots
crop_fig     = False # banded plots related cropped image in the main paper vs full image in the supp paper
Ncomp        = 5      # no. of binned fft components. Other option are not available\
uint16_dtype = True   # input image is uint16 or uint8 

if crop_fig:
    # -------------------------------------------------------------------
    # this option corresponds to banded plots of the cropped ROIs 
    # between normal resolution from FBP and SRGAN shown in the main paper.
    # cnn refers to SRGAN-based outputs
    # ---------------------------------------------------------------------
    if uint16_dtype: 
        gt_path  = '../paper_plots/crop_banded/crop_img_uint16/gt_000069.raw'
        cnn_path = '../paper_plots/crop_banded/crop_img_uint16/srgan_000069.raw'    
        gt_img   = io_func.raw_imread(gt_path, (90, 90), 'uint16')
        cnn_img  = io_func.raw_imread(cnn_path, (90, 90), 'uint16')
        out_path = '../paper_plots/crop_banded/Ncomp_'+ str(Ncomp)+ '_uint16/'
    else:
        gt_path  = '../paper_plots/crop_banded/crop_img_uint8_L_50_W_400/gt_000069.png'
        cnn_path = '../paper_plots/crop_banded/crop_img_uint8_L_50_W_400/srgan_000069.png'
        gt_img   = io_func.imageio_imread(gt_path)
        cnn_img  = io_func.imageio_imread(cnn_path)
        out_path = '../paper_plots/crop_banded/Ncomp_'+ str(Ncomp)+ '_uint8/'

else:
    # -----------------------------------------------------------------
    # this option corresponds to the banded plots of full images 
    # related to conventional artifacts showns in supplemental document.
    # cnn:  refers to missing-wedge-based outputs. 
    # dist: refers to distortion-based outputs
    # -----------------------------------------------------------------
    gt_path     = './data/L506_sh_fd_slice_32/mk_L506_tv_slice_32.raw'
    cnn_path    = './results/missing_wedge/input/L506_sh_slice_32_recon_theta_pm60_spac_2_uint16.raw'      # missing_wedge
    dist_path   = './results/distortion/input/L506_sh_slice_32_recon_theta_deg10_spac_0.5_uint16.raw' # distortion
    gt_img      = io_func.raw_imread(gt_path, (512, 512), 'uint16')
    cnn_img     = io_func.raw_imread(cnn_path, (512, 512), 'uint16')
    dist_img    = io_func.raw_imread(dist_path, (512, 512), 'uint16')
    out_path    = '../paper_plots/full_img_banded/Ncomp_'+ str(Ncomp)+ '_uint16/'

h, w    = gt_img.shape
r       = int(h/2)
    
fs_c1   = int(0.1*r) 
fs_c2   = int(0.25*r)
fs_c3   = int(0.5*r)
fs_c4   = int(0.75*r)
# -----------------------------------------------------------------------------------------------------------------------------------------
# Setting the display windows
# ------------------------------------------------------------------------------------------------------------------------------------------
if crop_fig: # cropped figure shown in the main paper --------------------------------------------------------------------------------------
    if uint16_dtype:
        gt_img     = pf.img_windowing(gt_img, windowing='soft')
        cnn_img    = pf.img_windowing(cnn_img, windowing='soft')
    win_arr    = ['None', 'None', 'None', 'None', 'None', 'None']# ['fk_ct_soft', 'fk_ct_soft', 'None', 'band_3', 'band_4', 'band_5']#
    binary_out = True # outputs only the banded filter as the top subplot otherwise yields filter convolved with FFT as the top row subplot
else: # banded plots corresponed to the full figure shown in the supp paper -------------------------------------------------------------------
    win_arr = ['artifact_lf', 'artifact_lf', 'None', 'band_3', 'band_4', 'band_5']
    dist_c1, dist_c2, dist_c3, dist_c4, dist_c5, dist_c_stack = get5_comp_of_img(dist_img, fs_c1, fs_c2, fs_c3, fs_c4, r, win_arr=win_arr) 
    binary_out = False # outputs only the banded filter as the top subplot otherwise yields filter convolved with FFT as the top row subplot

cnn_c1, cnn_c2, cnn_c3, cnn_c4, cnn_c5, cnn_c_stack                         = get5_comp_of_img(cnn_img, fs_c1, fs_c2, fs_c3, fs_c4, r, win_arr=win_arr) # missing-wedge-img/srgan-patch
gt_c1,  gt_c2,  gt_c3, gt_c4, gt_c5, gt_c_stack                             = get5_comp_of_img(gt_img, fs_c1, fs_c2, fs_c3, fs_c4, r, win_arr=win_arr)  # GT
gt_fft_c1, gt_fft_c2, gt_fft_c3, gt_fft_c4, gt_fft_c5, gt_fft_c_stack       = get5_fft_bands_of_img(gt_img, fs_c1, fs_c2, fs_c3, fs_c4, r) # GT FFT
cnn_fft_c1, cnn_fft_c2, cnn_fft_c3, cnn_fft_c4, cnn_fft_c5, cnn_fft_c_stack = get5_fft_bands_of_img(cnn_img, fs_c1, fs_c2, fs_c3, fs_c4, r, binary_out=binary_out)


if plot_fig:
    if crop_fig:
        # the three subplot rows correspond to  banded filters, srgan, gt rows --------------------------------------------------------------------------------
        band_stacks = np.stack((cnn_fft_c_stack.reshape(6, h, w), cnn_c_stack.reshape(6, h, w), gt_c_stack.reshape(6, h, w)), axis=0)
        print('shape of the subplots:', (band_stacks.reshape(18, h, w)).shape)
        pf.multi2dplots(3, 6, band_stacks.reshape(18, h, w), axis=0, passed_fig_att={'colorbar': False, 'figsize': [11, 6]}) 
    else:
        # fine details in band 3 through 5 do not appear properly due to variations in intensity between different plots ---------------------------------------
        # so arrange plot individually using save_fig below
        # the four subplot rows correspond to banded ffilters (convolved to gt), gt, missingwedge, distortion rows
        band_stacks               = np.stack((gt_fft_c_stack.reshape(6, h, w), gt_c_stack.reshape(6, h, w), cnn_c_stack.reshape(6, h, w), dist_c_stack.reshape(6, h, w)),axis=0)
        all_bands_in_arr          = (band_stacks.reshape(24, h, w))
        last_3_cols_arr           = np.array([3,4,5,9,10,11,15,16,17,21,22,23])
        last_col                  = np.array([5,11,17,23])
        last_3_cols_of_banded_arr = all_bands_in_arr[last_col, :, :]
        #out_banded_img_name = ('./banded_plots/full_fig/Ncomp_5_uint16_all_aw.eps')
        pf.multi2dplots(4, 6, all_bands_in_arr, axis=0, passed_fig_att={'colorbar': False, 'figsize': [8, 6]})#, 'out_path': out_banded_img_name})

if save_fig:
    if not os.path.isdir(out_path): os.makedirs(out_path, exist_ok=True)

    # multiplot-based subplots from python is sufficient for outputting frequency-wised banded outputs for the cropped figure
    # however for the full image, fine details are lost in the python-based subplots. Hence, for the full image all individual banded
    # outputs are saved as figures and manually arranged. 
    if crop_fig:
        pf.multi2dplots(3, 6, band_stacks.reshape(18, h, w), axis=0, passed_fig_att={'colorbar': False, 'figsize': [11, 6], 'out_path': out_path + 'indentation_all_bands.png'})
    else:
        # missing_wedge parts ----------------------------------------------
        cnn_c1    = utils.normalize_data_ab(0, 255, cnn_c1).astype('uint8')
        cnn_c2    = utils.normalize_data_ab(0, 255, cnn_c2).astype('uint8')
        cnn_c3    = utils.normalize_data_ab(0, 255, cnn_c3).astype('uint8')
        cnn_c4    = utils.normalize_data_ab(0, 255, cnn_c4).astype('uint8')
        cnn_c5    = utils.normalize_data_ab(0, 255, cnn_c5).astype('uint8')
        
        #normal resolution (gt) parts ----------------------------------------
        gt_c1     = utils.normalize_data_ab(0, 255, gt_c1).astype('uint8')
        gt_c2     = utils.normalize_data_ab(0, 255, gt_c2).astype('uint8')
        gt_c3     = utils.normalize_data_ab(0, 255, gt_c3).astype('uint8')
        gt_c4     = utils.normalize_data_ab(0, 255, gt_c4).astype('uint8')
        gt_c5     = utils.normalize_data_ab(0, 255, gt_c5).astype('uint8')

        # distortion (dist) part -----------------------------------------------
        dist_c1   = utils.normalize_data_ab(0, 255, dist_c1).astype('uint8')
        dist_c2   = utils.normalize_data_ab(0, 255, dist_c2).astype('uint8')
        dist_c3   = utils.normalize_data_ab(0, 255, dist_c3).astype('uint8')
        dist_c4   = utils.normalize_data_ab(0, 255, dist_c4).astype('uint8')
        dist_c5   = utils.normalize_data_ab(0, 255, dist_c5).astype('uint8')
        
        #gt fft parts ----------------------------------------------------------
        gt_fft_c1 = utils.normalize_data_ab(0, 255, gt_fft_c1).astype('uint8')
        gt_fft_c2 = utils.normalize_data_ab(0, 255, gt_fft_c2).astype('uint8')
        gt_fft_c3 = utils.normalize_data_ab(0, 255, gt_fft_c3).astype('uint8')
        gt_fft_c4 = utils.normalize_data_ab(0, 255, gt_fft_c4).astype('uint8')
        gt_fft_c5 = utils.normalize_data_ab(0, 255, gt_fft_c5).astype('uint8')
        gt_fft    = utils.normalize_data_ab(0, 255, gt_fft_c_stack[0]).astype('uint8')
        
        io_func.imsave(cnn_c1, path=out_path + 'mw_c1.png', svtype='original')
        io_func.imsave(cnn_c2, path=out_path + 'mw_c2.png', svtype='original')
        io_func.imsave(cnn_c3, path=out_path + 'mw_c3.png', svtype='original')
        io_func.imsave(cnn_c4, path=out_path + 'mw_c4.png', svtype='original')
        io_func.imsave(cnn_c5, path=out_path + 'mw_c5.png', svtype='original')

        io_func.imsave(dist_c1, path=out_path + 'dist_c1.png', svtype='original')
        io_func.imsave(dist_c2, path=out_path + 'dist_c2.png', svtype='original')
        io_func.imsave(dist_c3, path=out_path + 'dist_c3.png', svtype='original')
        io_func.imsave(dist_c4, path=out_path + 'dist_c4.png', svtype='original')
        io_func.imsave(dist_c5, path=out_path + 'dist_c5.png', svtype='original')

        io_func.imsave(gt_c1, path=out_path + 'gt_c1.png', svtype='original')
        io_func.imsave(gt_c2, path=out_path + 'gt_c2.png', svtype='original')
        io_func.imsave(gt_c3, path=out_path + 'gt_c3.png', svtype='original')
        io_func.imsave(gt_c4, path=out_path + 'gt_c4.png', svtype='original')
        io_func.imsave(gt_c5, path=out_path + 'gt_c5.png', svtype='original')
        
        io_func.imsave(gt_fft_c1, path=out_path + 'gt_fft_c1.png', svtype='original')
        io_func.imsave(gt_fft_c2, path=out_path + 'gt_fft_c2.png', svtype='original')
        io_func.imsave(gt_fft_c3, path=out_path + 'gt_fft_c3.png', svtype='original')
        io_func.imsave(gt_fft_c4, path=out_path + 'gt_fft_c4.png', svtype='original')
        io_func.imsave(gt_fft_c5, path=out_path + 'gt_fft_c5.png', svtype='original')
        io_func.imsave(gt_fft,    path=out_path + 'gt_fft.png',    svtype='original')


