import os
import utils
import numpy as np
import sys


import argparse
import mpi_utils 
from mpi4py import MPI
import random
import h5py


def sfrc_in_mpi(args):
	#----------- MPI declarations--------------------------
	comm  = MPI.COMM_WORLD
	rank  = comm.Get_rank()
	nproc = comm.Get_size()
	dest  = 0
	root  = 0
	dtype = args.dtype
	
	# -----------------------------------------------------
	#  Variables to be used for 
	#  mpi-based reduce Operations
	# -----------------------------------------------------
	partition_img_sum = np.zeros(1, dtype=dtype)
	# float declarations
	global_pre_norm_tar_min  = np.zeros(1, dtype=dtype)
	global_pre_norm_tar_max  = np.zeros(1, dtype=dtype)
	global_post_norm_tar_min = np.zeros(1, dtype=dtype)
	global_post_norm_tar_max = np.zeros(1, dtype=dtype)
	per_cz_fk                = np.zeros(1, dtype=dtype)
	tot_no_of_fk             = np.zeros(1, dtype=dtype)
	# dict diclarations
	partitioned_data = None

	# ------------------------------------
	# Read images &
	# Declare required variables 
	# & arrays to be broadcasted
	# from root to other ranks
	# -----------------------------------
	if rank==root:

		print('\n----------------------------------------')
		print('Command line arguements')
		print('----------------------------------------')
		for i in args.__dict__: print((i),':',args.__dict__[i])
		print('----------------------------------------')
		if args.img_format != 'dicom':
			print('')
			print('***********************WARNING*******************')
			print(' Images are not in dicom format.')
			print(' Ensure that the IMG format for input-targets pair and their sizes are accurately set')
			print(' in the function partition_read_n_sfrc_plot_n_calc in file mpi_utils.py.')
			print('************************************************')
			
		#--------------read all images in the input-target folder pair-----------------------
		all_input_paths, all_target_paths = mpi_utils.img_paths4rm_training_directory(args)

		#-------------------------------------------------------------------------------
		# No. of images (or images in a chunk) to be processed by each processor
		# Ensuring that N_images % nproc = 0
		#-------------------------------------------------------------------------------
		if (np.mod(len(all_target_paths), nproc)!=0):
			print('\n==>Total no. of image pairs to be compared is', len(all_target_paths), 'using', nproc, 'ranks.')
			print('==>Hence, removing last', np.mod(len(all_target_paths), nproc), 'image path(s)', end =' ')
			print('from the dataset such that the mod(N_images,nproc)=0')
			N_last_paths = np.mod(len(all_target_paths), nproc)
			all_target_paths = all_target_paths[:-N_last_paths]
			all_input_paths  = all_input_paths [:-N_last_paths]
		
		#--------------------------------------------
		# Dose blend option is not used in sfrc calculations.
		# Hence, the blending array is simply employed as a NULL
		# placeholder in all the rank distribution-based
		# functions in this package. 
		#--------------------------------------------
		if (args.dose_blend):
			blend_fact_arr = np.random.uniform(0.5,1.2,size=len(all_target_paths))
		else: 
			blend_fact_arr =np.zeros((len(all_target_paths),1), dtype=dtype)
		per_ip_fk_arr = np.zeros((len(all_target_paths),1))

		chunck = int(len(all_target_paths)/nproc) 
		if (len(all_target_paths)!=len(all_input_paths)):
			print('ERROR! Mismatch in the no. of DL/Reg vs. reference method-based images.')
			sys.exit()
		print("",len(all_target_paths),"DL/Reg and reference method-based image pairs are processed by", nproc, \
			"rank(s); \n with each rank's calculation distributed across", chunck, "chunks.")
	
		print('\n')
		print('===================================')
		print('input images from methods:')
		print('===================================')
		print(all_input_paths)

		print("\n\n")
		print('===================================')
		print('Reference image paths:')
		print('===================================')
		print(all_target_paths)

		if args.img_y_padding:
			print('')
			print('**************************************************************************************************')
			print('Ignore any warning that states:- RuntimeWarning: invalid value encountered in true_divide.')
			print('This is primarily due to nan values corresponding to patches filled with zeros after padding.')
			print('This error is trashed out automatically through in-build')
			print('air thresholding function applied before marking candidate patches')
			print('**************************************************************************************************')
		#-------------------------------------------creating output folder to save sfrc curves and fake ROIs as bounded subplots--------------------------------------------------------------------------------------------------------------
		if args.mtf_space:
			output_folder = args.output_folder + (args.patch_size) + '_lp__frc_thres_' + str(args.frc_threshold) + '_hn_' + str(args.apply_hann)[0] + '_bm_' + str(args.apply_bm3d)[0] + '_bf_' +  str(args.remove_ref_noise)[0] + '_mtfS_T' + '/'
		else:
			output_folder = args.output_folder + (args.patch_size) + '_lp_frc_thres_' + str(args.frc_threshold) + '_hn_' + str(args.apply_hann)[0] + '_bm_' + str(args.apply_bm3d)[0] + '_bf_' +  str(args.remove_ref_noise)[0] + '_frcS_T' + '/'
		if not os.path.isdir(output_folder): os.makedirs(output_folder)

		# creating output folder to save patched subplots
		if args.save_patched_subplots:
			output_patched_folder      = args.output_folder + (args.patch_size) + '_img_' + args.windowing + '_frcT_' + str(args.frc_threshold)+ '_halluT_' + str(args.ht) + '_bm_' + str(args.apply_bm3d)[0] + '_bf_' +  str(args.remove_ref_noise)[0]+'/'
			method_patched_folder_name = output_patched_folder + 'method/'
			ref_patched_folder_name    = output_patched_folder + 'ref/'
			if not os.path.isdir(method_patched_folder_name): os.makedirs(method_patched_folder_name)
			if not os.path.isdir(ref_patched_folder_name): os.makedirs(ref_patched_folder_name)
		else:
			output_patched_folder=None
		
		#---------------------------------------input-output variables for each rank that is initialized by root--------------------------------------------------------------------------------------------
		bcasted_input_data = {'all_target_paths': all_target_paths, 'all_input_paths': all_input_paths,\
							  'chunck': chunck, 'nproc':nproc, 'blend_fact_arr': blend_fact_arr, 'output_folder': output_folder,\
							  'output_patched_folder':output_patched_folder, 'tot_fk': tot_no_of_fk, 'cz_fk': per_cz_fk}

	else:
		bcasted_input_data  = None

	bcasted_input_data  = comm.bcast(bcasted_input_data, root=root)
	comm.Barrier()
	
	#----------------------sFRC operation ----------------------------------
	mpi_utils.partition_read_n_sfrc_plot_n_calc(args, bcasted_input_data, rank)
	comm.Barrier()