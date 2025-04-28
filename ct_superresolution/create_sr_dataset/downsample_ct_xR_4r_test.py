import sys
import numpy as np 
import os
from skimage.transform import rescale, resize, downscale_local_mean
import cv2
import glob_funcs as gf
import io_func as io


def main():
	'''Output both the folders, on selecting/tuning 
	images as well as testing images for sFRC.
	'''
	home_dir        = '/projects01/didsr-aiml/prabhat.kc'
	ua_ll_L067      = True # so that images only from 00030 through 00222 are used in the analysis
	sm_L067         = True # downsampling smooth vs. sharp test folder. image no 107 is removed from smooth folder folder such that it is consistent with sharp test images. 
	
	if sm_L067: #smooth
		sel_img_4r_L067 = [39, 69, 149, 189, 217] # images used to tune sFRC
		in_dir_name     = home_dir + '/common_data/LDGC/full_3mm/full_3mm_smooth/L067/full_3mm'
		out_dir_name    = home_dir + '/code/GitRepo/mpi_sfrc/ct_superresolution/data/test_sm_L067/ua_ll_3mm'

	else: #sharp
		sel_img_4r_L067 = [39, 69, 148, 188, 216] # last three images are one step below due to the fact that sh L067 in LDGC is missing 107 
		in_dir_name    = home_dir + '/common_data/LDGC/full_3mm/full_3mm_sharp/L067/full_3mm_sharp_sorted'
		out_dir_name   = home_dir + '/code/GitRepo/mpi_sfrc/ct_superresolution/data/test_sh_L067/ua_ll_3mm'

	scale 		   = 4
	input_img_type = 'dicom'
	in_dtype	   = 'uint16'
	out_dtype      = in_dtype
    
	hr_out_dir = os.path.join(out_dir_name + '_HR_test_x' + str(scale) + '/')
	lr_out_dir = os.path.join(out_dir_name + '_LR_test_x' + str(scale) +'/')
	
	sel_hr_out_dir = os.path.join(out_dir_name + '_HR_tune_x' + str(scale) + '/')
	sel_lr_out_dir = os.path.join(out_dir_name + '_LR_tune_x' + str(scale) +'/')
	
	if not os.path.isdir(hr_out_dir): os.makedirs(hr_out_dir)
	if not os.path.isdir(lr_out_dir): os.makedirs(lr_out_dir)
	if not os.path.isdir(sel_hr_out_dir): os.makedirs(sel_hr_out_dir)
	if not os.path.isdir(sel_lr_out_dir): os.makedirs(sel_lr_out_dir)
	
	img_names = io.getimages4rmdir(in_dir_name)
	

	if ua_ll_L067:
		sp = 30
		ep = len(img_names)
	else:
		sp = 0
		ep = len(img_names)
	
	for i in range(sp, ep):
		# removing image 107 from smooth folder
		if sm_L067==True and i==107:
			continue
		
		if input_img_type == 'dicom':
			image = gf.pydicom_imread(img_names[i])
		else:
			image = gf.imageio_imread(img_names[i])

		hr_image = gf.modcrop(image, scale)
		if len(hr_image.shape) == 3:
			h, w, _ = hr_image.shape
		else:
			h, w    = hr_image.shape
		#lr_image = resize(hr_image/img_max, (h / scale, w/ scale), anti_aliasing=True)
		#print(h, w)
		lr_image = cv2.resize(hr_image, (int(w/scale), int(h/scale)),interpolation=cv2.INTER_AREA)

		hr_image = hr_image.astype(out_dtype)
		lr_image = lr_image.astype(out_dtype)

		in_img_str  = img_names[i]
		out_img_str = in_img_str.split('/')[-1]
		out_img_str = out_img_str[:-4]
		
		if i in sel_img_4r_L067:
			io.imsave_raw(lr_image, sel_lr_out_dir + out_img_str + '.raw' )
			io.imsave_raw(hr_image, sel_hr_out_dir + out_img_str + '.raw')
		else:
			io.imsave_raw(lr_image, lr_out_dir + out_img_str + '.raw' )
			io.imsave_raw(hr_image, hr_out_dir + out_img_str + '.raw')

		print('zooming achived #:', out_img_str, 'HR shape:', hr_image.shape, 'LR shape:', lr_image.shape, \
		'HR range:', '(',np.min(hr_image),',', np.max(hr_image),')', 'LR range:', '(',np.min(lr_image),',', np.max(lr_image),')')
		
	print('=======================================================================================================')
	print('HR output folder:', hr_out_dir)
	print('LR output folder:', lr_out_dir)
	print('=======================================================================================================')
if __name__ == '__main__':
	main()



