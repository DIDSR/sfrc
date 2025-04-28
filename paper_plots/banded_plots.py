##############################################################
# @author: pkc 
#
# banded_plots.py
# ............
# includes codes to output different banded plots shown in the
# sfrc paper
# comment the line mpi.use('Agg') in the file src/plot_func.py
# to view plot as display

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
# --------------------------------------
# Display and saving figure option
# -----------------------------------------
plot_fig   = True   # Display plots
save_fig   = False  # save plots
crop_fig   = True   # banded plots related cropped image in the main paper vs full image in the supp paper
Ncomp      = 5      # no. of binned fft components. Other option is 3

if crop_fig:
    # -----------------------------------------------------------------
    # this option corresponds to banded plots of the cropped ROIs 
    # between FBP and SRGAN shown in the main paper.
    # cnn is pointed to SRGAN-based outputs
    # -----------------------------------------------------------------
    gt_path  = './plot2/crop_img_uint8_L_50_W_400/gt_000069.png'
    cnn_path = './plot2/crop_img_uint8_L_50_W_400/srgan_000069.png' #SRGAN outputs
    out_path = './plot2/crop_fig/Ncomp_'+ str(Ncomp)+ '/'
else:
    # -----------------------------------------------------------------
    # this option corresponds to the banded plots of full images 
    # related to conventional artifacts showns in supplemental document.
    # cnn is pointed to missing-wedge-based outputs. 
    # -----------------------------------------------------------------
    #gt_path  = '../nfk_artifacts/data/missing-wedge/gt_irt/mk_L291_tv_000163_uint16.tif'
    #cnn_path = '../nfk_artifacts/data/missing-wedge/input_irt/recon_theta_pm60_spac_2_uint16.tif'
    gt_path   = './plot2/full_img_uint8_L_1256_W_780/mk_L291_tv_000163_L_1286_W_780_2_uint8.png'
    cnn_path  = './plot2/full_img_uint8_L_1256_W_780/recon_theta_pm60_spac_2_L_1286_W_780_2_uint8.png' # missing_wedge
    dist_path = './plot2/full_img_uint8_L_1256_W_780/recon_theta_deg10_spac_0.5_L_1286_W_780_2_uint8.png' # distortion
    out_path  = './plot2/full_fig/Ncomp_'+ str(Ncomp)+ '_uint8/'

if not os.path.isdir(out_path): os.makedirs(out_path, )
if not crop_fig: dist_img = io_func.imageio_imread(dist_path)

gt_img  = io_func.imageio_imread(gt_path)
cnn_img = io_func.imageio_imread(cnn_path)
h, w    = gt_img.shape
r       = int(h/2)

if Ncomp == 3:
    fs_c1  = int(0.1*r) 
    fs_c2  = int(0.25*r)
    
    cnn_c1, cnn_c2, cnn_c3, cnn_c_stack = get3_comp_of_img(cnn_img, fs_c1, fs_c2, r)
    gt_c1,  gt_c2,  gt_c3,  gt_c_stack  = get3_comp_of_img(gt_img, fs_c1, fs_c2, r)
    
    cnn_fft_c1, cnn_fft_c2, cnn_fft_c3, cnn_fft_c_stack = get3_fft_bands_of_img(cnn_img, fs_c1, fs_c2, r)
    
    if plot_fig:
        pf.multi2dplots(1, 4, cnn_c_stack, axis=0, passed_fig_att={'colorbar': False})
        pf.multi2dplots(1, 4, cnn_fft_c_stack, axis=0, passed_fig_att={'colorbar': False})
        pf.plot2dlayers(gt_img)
    
    if save_fig:
        if not os.path.isdir(out_path): os.makedirs(out_path, exist_ok=True)
        # cnn part
        cnn_c1 = utils.normalize_data_ab(0, 255, cnn_c1).astype('uint8')
        cnn_c2 = utils.normalize_data_ab(0, 255, cnn_c2).astype('uint8')
        cnn_c3 = utils.normalize_data_ab(0, 255, cnn_c3).astype('uint8')
        
        #gt parts
        gt_c1 = utils.normalize_data_ab(0, 255, gt_c1).astype('uint8')
        gt_c2 = utils.normalize_data_ab(0, 255, gt_c2).astype('uint8')
        gt_c3 = utils.normalize_data_ab(0, 255, gt_c3).astype('uint8')
        #cnn fft parts
        cnn_fft_c1 = utils.normalize_data_ab(0, 255, cnn_fft_c1).astype('uint8')
        cnn_fft_c2 = utils.normalize_data_ab(0, 255, cnn_fft_c2).astype('uint8')
        cnn_fft_c3 = utils.normalize_data_ab(0, 255, cnn_fft_c3).astype('uint8')
        cnn_fft    = utils.normalize_data_ab(0, 255, cnn_fft_c_stack[0]).astype('uint8')
        io_func.imsave(cnn_c1, path=out_path + 'cnn_c1.png', svtype='original')
        io_func.imsave(cnn_c2, path=out_path + 'cnn_c2.png', svtype='original')
        io_func.imsave(cnn_c3, path=out_path + 'cnn_c3.png', svtype='original')
    
        io_func.imsave(gt_c1, path=out_path + 'gt_c1.png', svtype='original')
        io_func.imsave(gt_c2, path=out_path + 'gt_c2.png', svtype='original')
        io_func.imsave(gt_c3, path=out_path + 'gt_c3.png', svtype='original')
        
        io_func.imsave(cnn_fft_c1, path=out_path + 'cnn_fft_c1.png', svtype='original')
        io_func.imsave(cnn_fft_c2, path=out_path + 'cnn_fft_c2.png', svtype='original')
        io_func.imsave(cnn_fft_c3, path=out_path + 'cnn_fft_c3.png', svtype='original')
        io_func.imsave(cnn_fft, path=out_path    + 'cnn_fft.png', svtype='original')

elif Ncomp==5:
    fs_c1  = int(0.1*r) 
    fs_c2  = int(0.25*r)
    fs_c3  = int(0.5*r)
    fs_c4  = int(0.75*r)
    
    cnn_c1, cnn_c2, cnn_c3, cnn_c4, cnn_c5, cnn_c_stack = get5_comp_of_img(cnn_img, fs_c1, fs_c2, fs_c3, fs_c4, r) # missing-wedge-img/srgan-patch
    gt_c1,  gt_c2,  gt_c3, gt_c4, gt_c5, gt_c_stack     = get5_comp_of_img(gt_img, fs_c1, fs_c2, fs_c3, fs_c4, r)
    cnn_fft_c1, cnn_fft_c2, cnn_fft_c3, cnn_fft_c4, cnn_fft_c5, cnn_fft_c_stack = get5_fft_bands_of_img(cnn_img, fs_c1, fs_c2, fs_c3, fs_c4, r)
    
    if not crop_fig:
        dist_c1, dist_c2, dist_c3, dist_c4, dist_c5, dist_c_stack = get5_comp_of_img(dist_img, fs_c1, fs_c2, fs_c3, fs_c4, r) 

    if plot_fig:
        if crop_fig:
            # the three rows correspond to FFT of SRGAN, SRGAN and FBP
            band_stacks = np.stack((cnn_fft_c_stack.reshape(6, h, w), cnn_c_stack.reshape(6, h, w), gt_c_stack.reshape(6, h, w)), axis=0)
            print('shape of the subplots:', (band_stacks.reshape(18, h, w)).shape)
            pf.multi2dplots(3, 6, band_stacks.reshape(18, h, w), axis=0, passed_fig_att={'colorbar': False, 'figsize': [8, 6]})#, 'out_path': out_path + 'all_bands.png'})
        else:
            band_stacks = np.stack((cnn_fft_c_stack.reshape(6, h, w), cnn_c_stack.reshape(6, h, w), dist_c_stack.reshape(6, h, w), gt_c_stack.reshape(6, h, w)), axis=0)
            print('shape of the subplots:', (band_stacks.reshape(24, h, w)).shape)
            pf.multi2dplots(4, 6, band_stacks.reshape(24, h, w), axis=0, passed_fig_att={'colorbar': False, 'figsize': [8, 6]})#, 'out_path': out_path + 'all_bands.png'})
    
    if save_fig:
        if not os.path.isdir(out_path): os.makedirs(out_path, exist_ok=True)
        # cnn part
        cnn_c1 = utils.normalize_data_ab(0, 255, cnn_c1).astype('uint8')
        cnn_c2 = utils.normalize_data_ab(0, 255, cnn_c2).astype('uint8')
        cnn_c3 = utils.normalize_data_ab(0, 255, cnn_c3).astype('uint8')
        cnn_c4 = utils.normalize_data_ab(0, 255, cnn_c4).astype('uint8')
        cnn_c5 = utils.normalize_data_ab(0, 255, cnn_c5).astype('uint8')
        
        #gt parts
        gt_c1 = utils.normalize_data_ab(0, 255, gt_c1).astype('uint8')
        gt_c2 = utils.normalize_data_ab(0, 255, gt_c2).astype('uint8')
        gt_c3 = utils.normalize_data_ab(0, 255, gt_c3).astype('uint8')
        gt_c4 = utils.normalize_data_ab(0, 255, gt_c4).astype('uint8')
        gt_c5 = utils.normalize_data_ab(0, 255, gt_c5).astype('uint8')
        
        #cnn fft parts
        cnn_fft_c1 = utils.normalize_data_ab(0, 255, cnn_fft_c1).astype('uint8')
        cnn_fft_c2 = utils.normalize_data_ab(0, 255, cnn_fft_c2).astype('uint8')
        cnn_fft_c3 = utils.normalize_data_ab(0, 255, cnn_fft_c3).astype('uint8')
        cnn_fft_c4 = utils.normalize_data_ab(0, 255, cnn_fft_c4).astype('uint8')
        cnn_fft_c5 = utils.normalize_data_ab(0, 255, cnn_fft_c5).astype('uint8')
        cnn_fft    = utils.normalize_data_ab(0, 255, cnn_fft_c_stack[0]).astype('uint8')
        
        io_func.imsave(cnn_c1, path=out_path + 'cnn_c1.png', svtype='original')
        io_func.imsave(cnn_c2, path=out_path + 'cnn_c2.png', svtype='original')
        io_func.imsave(cnn_c3, path=out_path + 'cnn_c3.png', svtype='original')
        io_func.imsave(cnn_c4, path=out_path + 'cnn_c4.png', svtype='original')
        io_func.imsave(cnn_c5, path=out_path + 'cnn_c5.png', svtype='original')
    
        io_func.imsave(gt_c1, path=out_path + 'gt_c1.png', svtype='original')
        io_func.imsave(gt_c2, path=out_path + 'gt_c2.png', svtype='original')
        io_func.imsave(gt_c3, path=out_path + 'gt_c3.png', svtype='original')
        io_func.imsave(gt_c4, path=out_path + 'gt_c4.png', svtype='original')
        io_func.imsave(gt_c5, path=out_path + 'gt_c5.png', svtype='original')
        
        io_func.imsave(cnn_fft_c1, path=out_path + 'cnn_fft_c1.png', svtype='original')
        io_func.imsave(cnn_fft_c2, path=out_path + 'cnn_fft_c2.png', svtype='original')
        io_func.imsave(cnn_fft_c3, path=out_path + 'cnn_fft_c3.png', svtype='original')
        io_func.imsave(cnn_fft_c4, path=out_path + 'cnn_fft_c4.png', svtype='original')
        io_func.imsave(cnn_fft_c5, path=out_path + 'cnn_fft_c5.png', svtype='original')
        io_func.imsave(cnn_fft, path=out_path    + 'cnn_fft.png', svtype='original')
else:
    sys.exit('the only available options are 3 and 5')

