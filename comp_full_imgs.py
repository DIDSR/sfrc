# source /home/prabhat.kc/anaconda3/base_env.sh
# source /home/prabhat.kc/anaconda3/horovod_sm80_env.sh
# cd /projects01/didsr-aiml/prabhat.kc/code/measure_hallucination/exp2_frc

import sys
import os 

import frc_utils as frc_util
import secondary_utils as su 
import argparse
import io_func
import utils
import plot_func as pf
from skimage.metrics import structural_similarity as compare_ssim
import quant_util
import numpy as np
# FRC inputs
inside_square	= True
anaRing			= True
threshold       ='0.75' # '0.5'
apply_hann 		= True
disp_img 		= False
case_study      = 'MRI' # CT or MRI
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
# get io path info 
# when using os.join.path do not start second string with "/"
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------
if case_study=='CT': 
	Nx 		   = 256
	dtype 	   ='uint16'
	main_path  = '/projects01/didsr-aiml/prabhat.kc/code/mpi_sfrc/raw_data/'
	gt_folder  = os.path.join(main_path, 'L067/sp4_gt_x4')
	cnn_folder = os.path.join(main_path, 'L067/sp4_cnn_x4')
	bic_folder = os.path.join(main_path, 'L067/sp4_bc_x4')
	
	output_folder = './results/sp4_full_img/thres_' + str(threshold) + '_hann_' + str(apply_hann)[0] + '/'
	output_cnn_folder = output_folder + 'srgan/'
	output_bic_folder = output_folder + 'bic/'	
elif case_study=='MRI':
	#Nx         = 320
	#dtype      = 'uint16'
	second_comp_str ='bp' #'plstv' or 'bp'
	main_path  = '/projects01/didsr-aiml/prabhat.kc/code/mpi_sfrc/raw_data/mri/ood_test/'
	gt_folder  = os.path.join(main_path, 'masked_gt_ood_png_b')
	cnn_folder = os.path.join(main_path, 'masked_unet_recon_ood_png') # is unet
	bic_folder = os.path.join(main_path, 'masked_bp_acc1_ood_png') # is plstv
	
	output_folder = './results/mri/full_img_frc_' + str(threshold) + '_hann_' + str(apply_hann)[0] + '/'
	output_cnn_folder = output_folder + 'unet/'
	output_bic_folder = output_folder + second_comp_str + '_acc3/'
	cnn_psnr_arr, cnn_ssim_arr = [], []
	bic_psnr_arr, bic_ssim_arr = [], []
else:
	print("case study not defined")
	sys.exit()

if not os.path.isdir(output_cnn_folder): os.makedirs(output_cnn_folder)
if not os.path.isdir(output_bic_folder): os.makedirs(output_bic_folder)

gt_imgs  = io_func.getimages4rmdir(gt_folder)
cnn_imgs = io_func.getimages4rmdir(cnn_folder)
bic_imgs = io_func.getimages4rmdir(bic_folder)
Nimgs    = len(gt_imgs)
print(gt_imgs)
print('----')
print(cnn_imgs)
print('----')
print(bic_imgs)
print('----')

for i in range(Nimgs):
	if case_study=='CT':
		gt  = io_func.raw_imread(gt_imgs[i], (Nx, Nx), dtype)
		cnn = io_func.raw_imread(cnn_imgs[i], (Nx, Nx), dtype)	
		bic = io_func.raw_imread(bic_imgs[i], (Nx, Nx), dtype)		
		img_str = gt_imgs[i]
		img_str = img_str.split('/')[-1]
		img_no  = img_str.split('.')[-2]
		cnn_fname = output_cnn_folder + 'cnn_' + img_no + '.png'
		bic_fname = output_bic_folder + 'bic_' + img_no + '.png'

	elif case_study=='MRI':
		# stack_arr = []
		# read images
		gt   = io_func.imageio_imread(gt_imgs[i]).astype('float64')
		cnn  = io_func.imageio_imread(cnn_imgs[i]).astype('float64')
		bic  = io_func.imageio_imread(bic_imgs[i]).astype('float64')
		h, w = gt.shape
		
		# fnames/foldernames for outputs
		img_str   = gt_imgs[i]
		img_str   = img_str.split('/')[-1]
		img_no    = img_str.split('.')[-2]
		cnn_fname = output_cnn_folder + 'unet_' + img_no + '.png'
		bic_fname = output_bic_folder + second_comp_str +'_' + img_no + '.png'
		
		# global metrics-based comparisions
		cnn_max, cnn_min = max(np.max(gt), np.max(cnn)), min(np.min(gt), np.min(cnn))
		cnn_psnr = quant_util.psnr(gt, cnn, cnn_max)
		# cnn_ssim = compare_ssim(cnn.reshape(h, w, 1), gt.reshape(h, w, 1), multichannel=True, data_range=(cnn_max-cnn_min))
		cnn_ssim = compare_ssim(cnn, gt, multichannel=False, data_range=(cnn_max-cnn_min))
		cnn_psnr_arr.append(cnn_psnr)
		cnn_ssim_arr.append(cnn_ssim)
		
		bic_max, bic_min = max(np.max(gt), np.max(bic)), min(np.min(gt), np.min(bic))
		bic_psnr = quant_util.psnr(gt, bic, bic_max)
		# bic_ssim = compare_ssim(bic.reshape(h, w, 1), gt.reshape(h, w, 1), multichannel=True, data_range=(bic_max-bic_min))
		bic_ssim = compare_ssim(bic, gt, multichannel=False, data_range=(bic_max-bic_min))
		bic_psnr_arr.append(bic_psnr)
		bic_ssim_arr.append(bic_ssim)

		stack_arr = np.stack((gt, cnn, bic), axis=2)
		#pf.multi2dplots(1, 3, stack_arr, 2)

		print("IMG: %s || avg CNN [PSNR: %.4f, SSIM: %.4f] || avg %s [ PSNR: %.4f, SSIM: %.4f] || [ MAX: %.4f, MIN: %.4f]"\
                %(img_str, cnn_psnr, cnn_ssim, second_comp_str, bic_psnr, bic_ssim, cnn_max, cnn_min))
	else:
		print("case study not properly defined") 
		sys.exit()
	n_gt_img  = utils.normalize_data_ab(0.0, 1.0, gt)
	n_cnn_img = utils.normalize_data_ab(0.0, 1.0, cnn)
	n_bic_img = utils.normalize_data_ab(0.0, 1.0, bic)

	if apply_hann:
		n_gt_img = frc_util.apply_hanning_2d(n_gt_img)
		n_cnn_img = frc_util.apply_hanning_2d(n_cnn_img)
		n_bic_img = frc_util.apply_hanning_2d(n_bic_img)

	xc, corr_cnn, xt, thres_val = frc_util.FRC(n_gt_img, n_cnn_img, thresholding=threshold, inscribed_rings=inside_square, analytical_arc_based=anaRing)
	pf.plot_n_save_img(threshold=threshold, fx_coord=xc, frc_val=corr_cnn, tx_coord=xt, thres_val=thres_val, save_img=True, output_img_name=cnn_fname, display_img=disp_img)

	xc, corr_bic, xt, thres_val = frc_util.FRC(n_gt_img, n_bic_img, thresholding=threshold, inscribed_rings=inside_square, analytical_arc_based=anaRing)
	pf.plot_n_save_img(threshold=threshold, fx_coord=xc, frc_val=corr_bic, tx_coord=xt, thres_val=thres_val, save_img=True, output_img_name=bic_fname, display_img=disp_img)
	print("%3d/%3d done" %(i+1, Nimgs), end ='\r')

print('\n')
if case_study=='MRI':
	print("avg CNN (std) [PSNR: %.4f (%.4f), SSIM: %.4f (%.4f)] \navg %s  (std) [PSNR: %.4f (%.4f), SSIM: %.4f (%.4f)]" % 
		(np.mean(cnn_psnr_arr), np.std(cnn_psnr_arr), np.mean(cnn_ssim_arr), np.std(cnn_ssim_arr), second_comp_str, \
		np.mean(bic_psnr_arr), np.std(bic_psnr_arr), np.mean(bic_ssim_arr), np.std(bic_ssim_arr)))

# acc1
# avg CNN (std) [PSNR: 23.0980 (1.5157), SSIM: 0.3668 (0.0602)]
# avg BP  (std) [PSNR: 22.4866 (1.2652), SSIM: 0.3920 (0.0645)]

#acc2
#avg CNN (std) [PSNR: 23.0980 (1.5157), SSIM: 0.3668 (0.0602)]
#avg BP  (std) [PSNR: 23.2629 (1.8139), SSIM: 0.3801 (0.0632)]
#avg CNN (std) [PSNR: 23.0980 (1.5157), SSIM: 0.3668 (0.0602)]
#avg bp  (std) [PSNR: 22.6472 (1.3717), SSIM: 0.3363 (0.0559)]

