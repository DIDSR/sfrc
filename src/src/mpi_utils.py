import sys
import os
import numpy as np
import glob
from . import utils

import random
import bm3d
from . import io_func
from . import plot_func as pf
from skimage.restoration import denoise_bilateral
from mpi4py import MPI

def img_paths4rm_training_directory(args):
	"""
	Returns paths of input-target image pairs.
	It is considered that the input and target image files
	are stored inside the same folder called "input_folder"
	and vary by tag/folder_name input_gen_folder and target_gen_folder.
	
	input
	-----
	args : parser.parse_ags() from the command line. The following arguments
	       are used in this function
	       input_gen_folder : string tag/foldername where input images are stored
	       target_gen_folder: string tag/foldername where input images are stored
	       multi_patients   : (bool) in case the input-target folders are subfolders
	                          stored inside different patient folders. 
	       random_N         : (bool) used as developmental or debugging tool to randomly
	                          output couple of input-target pairs instead of all the images
	                          stored in the input-target folder. 
	output
	------
	two arrays with input and target
	path files with filepath names
	assorted in the ascending order. 

	"""
	if args.multi_patients: all_dir_paths = sorted(glob.glob(args.input_folder + '/*/'))
	else:					all_dir_paths = sorted(glob.glob(args.input_folder))

	all_input_paths, all_target_paths = [], []
	random_ind = None

	for dir_paths in all_dir_paths:
		if args.random_N: random_ind = utils.get_sorted_random_ind(os.path.join(dir_paths, args.input_gen_folder), args.N_rand_imgs)

		in_paths     = utils.getimages4rmdir(os.path.join(dir_paths, args.input_gen_folder), random_ind)
		all_input_paths.extend(in_paths)
		target_paths = utils.getimages4rmdir(os.path.join(dir_paths, args.target_gen_folder), random_ind)
		all_target_paths.extend(target_paths)
	return (np.asarray(all_input_paths), np.asarray(all_target_paths))

def partition_read_n_sfrc_plot_n_calc(args, bcasted_input_data, pid):
	"""
	"""
	comm                  = MPI.COMM_WORLD
	chunck_sz             = bcasted_input_data['chunck']
	all_input_paths       = bcasted_input_data['all_input_paths']
	all_target_paths      = bcasted_input_data['all_target_paths']
	nproc                 = bcasted_input_data['nproc']
	blend_fact_arr        = bcasted_input_data['blend_fact_arr']
	output_folder         = bcasted_input_data['output_folder']
	output_patched_folder = bcasted_input_data['output_patched_folder']
	tot_no_of_fk          = bcasted_input_data['tot_fk']
	per_cz_fk             = bcasted_input_data['cz_fk']
	# partition trackers to transfer
	pre_norm_tar_min, pre_norm_tar_max   = [], []
	post_norm_tar_min, post_norm_tar_max = [], []	
	
	for j in range(chunck_sz):
		# where chunk_sz = Nimgs/Nprocesses
		if args.img_format == 'dicom':
			input_image  = io_func.pydicom_imread(all_input_paths[pid*chunck_sz+j])
			target_image = io_func.pydicom_imread(all_target_paths[pid*chunck_sz+j])
		elif args.img_format == 'raw':
			input_image  = io_func.raw_imread(all_input_paths[pid*chunck_sz+j], (args.rNy, args.rNx), args.in_dtype)
			target_image = io_func.raw_imread(all_target_paths[pid*chunck_sz+j], (args.rNy, args.rNx), args.in_dtype)
			if args.img_y_padding:
				input_image  = np.pad(input_image,  ((0, args.rNx-args.rNy), (0, 0)))
				target_image = np.pad(target_image, ((0, args.rNx-args.rNy), (0, 0)))
		elif (args.img_format == 'tif' or args.img_format == 'png'):
			input_image  = io_func.imageio_imread(all_input_paths[pid*chunck_sz+j]).astype(args.dtype)
			target_image = io_func.imageio_imread(all_target_paths[pid*chunck_sz+j]).astype(args.dtype)
			#input_image  = utils.normalize_data_ab(0, args.img_list_max_val, input_image)
			#target_image = utils.normalize_data_ab(0, args.img_list_max_val, target_image)
		else:
			if(pid==0):
				print('ERROR! No read function for the specified image type:', args.img_format)
				print('ADD the required read function inside partition_read_normalize_n_augment in file mpi_utils.py')
				sys.exit()

		sp = target_image.shape

		# --------------------------------
		# Data channels & precision setup
		#---------------------------------
		if len(sp)==3:
			if(pid==0 and j==0): 
				print('\n==>Here target images have 3 colored channels but for training purposes we are only taking the first channel')
			target_image = (target_image[:, :, 0])
		
		if(pid==0 and j==0): 
			print('==> Here images from input paths is of type', args.in_dtype, end='.')
			print(' And is assigned as', (target_image.astype(args.dtype)).dtype,'for MPI-based calculations.')
			print('==> DL/regularization method-based images are sampled from', args.input_gen_folder, 'and are sized', input_image.shape)
			print('==> Reference method-based images are sampled from', args.target_gen_folder, 'and are sized', target_image.shape)
	
		target_image = target_image.astype(args.dtype)
		input_image  = input_image.astype(args.dtype)
		# dummy place holder to for air thresholding
		if(args.air_threshold): target_image_un = target_image
		# get blending factor 
		# in case of no dodse augmentation blending factor is simply 0
		blend_factor=blend_fact_arr[pid*chunck_sz+j]
			#print(pid, chunck, blend_factor)
		if (pid==0 and j==0): 
			print('')
			print("    Method-based images are outputs from a new DL/regularization-based method.")
			print("    Reference images are outputs from  standard-of-care method.")
		
		# -----------------------------------------------------------------
		# Data normalization & augmentation & air-thresholding
		# --------------------------------------------------------------
		pre_norm_tar_min.append(np.min(target_image)); pre_norm_tar_max.append(np.max(target_image))
		input_image, target_image = utils.img_pair_normalization(input_image, target_image, args.normalization_type)

		if args.apply_bm3d:
			#---------------before applying bm3d both images must be normalized-------------------------------------
			ref_max, ref_min            = np.max(target_image), np.min(target_image)
			me_max,  me_min             = np.max(input_image), np.min(input_image)
			binput_image, btarget_image = utils.img_pair_normalization(input_image, target_image, 'unity_independent')

			btarget_image               = bm3d.bm3d(btarget_image, sigma_psd=0.05, stage_arg=bm3d.BM3DStages.ALL_STAGES)
			binput_image                = bm3d.bm3d(binput_image, sigma_psd=0.05, stage_arg=bm3d.BM3DStages.ALL_STAGES)
			
			target_image                = utils.normalize_data_ab(ref_min, ref_max, btarget_image)
			input_image                 = utils.normalize_data_ab(me_min, me_max, binput_image)
		
		# -------------------using bilateral filtering to remove gently remove noise --------------------------------
		if args.remove_ref_noise:
			ref_max, ref_min            = np.max(target_image), np.min(target_image)
			_, gtarget_image            = utils.img_pair_normalization(input_image, target_image, 'unity_independent')
			gtarget_image               = denoise_bilateral(gtarget_image, win_size=7, sigma_color=0.02, sigma_spatial=5)
			target_image                = utils.normalize_data_ab(ref_min, ref_max, gtarget_image)

		post_norm_tar_min.append(np.min(target_image)); post_norm_tar_max.append(np.max(target_image))
		if(args.air_threshold): t_input_patch, t_target_patch = augment_n_return_patch(args, input_image, target_image, j, pid, blend_factor, target_image_un)
		else:					t_input_patch, t_target_patch = augment_n_return_patch(args, input_image, target_image, j, pid, blend_factor)

		# -------------------------------------------------------------------------------------------
		# send the patches generated/normalized/thresholded/edge removed to frc calculation 
		# ----------------------------------------------------------------------------------------------
		per_ip_fk = utils.patchwise_sfrc(args, t_input_patch, t_target_patch, j, pid, chunck_sz, all_target_paths[pid*chunck_sz+j], output_folder, output_patched_folder)
		comm.Reduce(per_ip_fk.astype(args.dtype), per_cz_fk, op=MPI.SUM, root=0)
		comm.Barrier()

		if(pid==0): 
			print("=========================================")
			print(j+1, ' of ', chunck_sz, 'chunks processed.  with', per_cz_fk, 'fakes')
			tot_no_of_fk = tot_no_of_fk + per_cz_fk
			print("=========================================")
		comm.Barrier()
	if(pid==0): 
		print('')
		print('--------------------------------------------------------------------------------------------------')
		print('total number of', args.patch_size,'sized fake patches across all the input images:', tot_no_of_fk)
		print('the sFRC curves and bounded box-based subplots as fakes ROIs are stored in:', output_folder)
		print('--------------------------------------------------------------------------------------------------')
		print('')

def augment_n_return_patch(args, input_image, target_image, i, pid, blend_factor, target_image_un=None):
	"""
	augmentation part is turned off and only patching part is 
	executed to extract patches from a given input-target image
	pairs in a distributed fashion using mpi
	here i is index within a chunk. Eg if a rank is processing 4 images 
	then i = 0, 1, 2, 3
	"""
	if args.ds_augment:
		input_aug_images  = utils.downsample_4r_augmentation(input_image)
		target_aug_images = utils.downsample_4r_augmentation(target_image)
		if (args.air_threshold): target_un_aug_images = utils.downsample_4r_augmentation(target_image_un)
		if (i==0 and pid==0): 
			print("\n==>Downscale based data augmentation is PERFORMED ...")
			print("   Also, each input-target image pair is downscaled by", len(input_aug_images)-1,"different scaling factors ...")
			print("   due to downscale based augmentation")
	else:
		h, w = input_image.shape
		input_aug_images  = np.reshape(input_image, (1, h, w))
		target_aug_images = np.reshape(target_image, (1, h, w))
		if (args.air_threshold): target_un_aug_images = np.reshape(target_image_un, (1, h, w))
		if(i==0 and pid==0): print("\n==>Augmentation is NoT PERFORMED")

	# declaring null array to append patches from augmented input & label later
	each_img_input_patch = np.empty([0, args.input_size, args.input_size, 1], dtype=args.dtype)
	each_img_target_patch = np.empty([0, args.label_size, args.label_size, 1], dtype=args.dtype)
	
	# Now working on each augmented images
	for p in range(len(input_aug_images)):
		label_ = (target_aug_images[p])
		input_ = (input_aug_images[p])
		
		label_ = utils.modcrop(target_aug_images[p], args.scale)
		input_ = utils.modcrop(input_aug_images[p], args.scale)
		if (args.air_threshold):un_label_ = target_un_aug_images[p]

		if args.scale ==1: input_ = input_
		else:			   input_ = utils.interpolation_lr(input_, args.scale)

		if args.blurr_n_noise: cinput_ = utils.add_blurr_n_noise(input_, seed[i])
		else:				   cinput_ = input_

		sub_input, sub_label = utils.overlap_based_sub_images(args, cinput_, label_)
		
		if(args.air_threshold):
			_, sub_label_un      = utils.overlap_based_sub_images(args, cinput_, un_label_) #cinput_ doesnot matter here
			sub_input, sub_label = utils.air_thresholding(args, sub_input, sub_label, sub_label_un)
		
		augmented_input, augmented_label = sub_input, sub_label
		if(args.rot_augment): augmented_input, augmented_label = utils.rotation_based_augmentation(args, augmented_input, augmented_label)
		if(args.dose_blend):  augmented_input, augmented_label = utils.dose_blending_augmentation(args, augmented_input, augmented_label, blend_factor)
		each_img_input_patch  = np.append(each_img_input_patch, augmented_input, axis=0)
		each_img_target_patch = np.append(each_img_target_patch, augmented_label, axis=0)

	'''
	if(i==0):# and pid ==1):
		window = 0
		lr_N = len(each_img_input_patch)
		rand_num=random.sample(range(lr_N-window), 1) # a single number
		s_ind  = 0 #rand_num[0]
		e_ind  = s_ind+window
		print('shape of first img from processor', pid, ':', each_img_input_patch.shape, each_img_target_patch.shape, each_img_input_patch.dtype, each_img_target_patch.dtype)
		# pf.multi2dplots(4, 8, each_img_input_patch[s_ind:e_ind, :, :, 0], 0, passed_fig_att = {"colorbar": False, "figsize":[4*args.scale, 4*args.scale]})
		# pf.multi2dplots(4, 8, each_img_target_patch[s_ind:e_ind, :, :, 0], 0, passed_fig_att = {"colorbar": False, "figsize":[4*args.scale, 4*args.scale]})
		# pf.dict_plot_of_2d_arr(1, 1, each_img_input_patch[s_ind:e_ind, :, :, 0], save_plot=False, disp_plot=True, output_path='', plt_title=str(pid))
		# pf.dict_plot_of_2d_arr(1, 1, each_img_target_patch[s_ind:e_ind, :, :, 0], save_plot=False, disp_plot=True, output_path='', plt_title=str(pid))
		# for validation check with patch 512
		# pf.plot2dlayers(each_img_input_patch[0, :, :, 0], title=str(pid)+' input')
		# pf.plot2dlayers(each_img_target_patch[0, :, :, 0], title=str(pid)+ ' target')
	# sys.exit()
	'''
	return(each_img_input_patch, each_img_target_patch)
	
def arrtobuff(arr):
	"""
	converts a higher dimensional array (2D or 3D or 4D) 
	into a rastered 1d array
	""" 
	buff = arr.ravel()
	return(buff)

def bufftoarr(buff, tot_element_count, ph, pw, pc):
	"""
	reshapes a 1d array to a higher dimensional 4D array
	based on the sizes provided as input
	""" 
	pz = int(tot_element_count/(ph*pw))
	arr = buff.reshape(pz, ph, pw, pc)
	return(arr)
