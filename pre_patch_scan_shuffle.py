import argparse
import sys
import mpi_utils
import numpy as np
import os
import shutil

# Command line arguments
parser = argparse.ArgumentParser(description='shuffling scans from all patients before using patch code')
parser.add_argument('--input-folder',  type=str, required=True, help='directory name containing training images')
parser.add_argument('--output-folder',  type=str, required=True, help='new directory name to containing randomized scans for all patients')
parser.add_argument('--input-gen-folder', type=str, default='quarter_3mm', help="folder name containing noisy (input) measurements")
parser.add_argument('--target-gen-folder', type=str, default='full_3mm', help="folder name containing clean (target) measurements")
parser.add_argument('--file-extension', type=str, default='.IMA', help="extension name of training images")
parser.add_argument('--nsplit', type=int, default=1, help="no of splits of the output folder")

args = parser.parse_args()

def rename_n_copy_file(shuff_paths, ind, final_dest, hold_fld, file_ext):
	init_path                     = shuff_paths[ind]
	init_path_split               = init_path.split('/')
	init_fn                       = init_path_split[-1]
	# first move to place holder
	shutil.move(init_path, hold_fld)
	# rename the file inside place holder 
	new_fn = hold_fld + '/' + str(ind) + file_ext
	os.rename(os.path.join(hold_fld, init_fn), new_fn)
	# copy the rename input to its final destination
	shutil.move(new_fn, final_dest)

# args.input_folder = 'raw_data_mixed'
# args.output_folder= 'raw_data_randomize'
args.random_N = None

input_folder_cp  = args.input_folder + '_cp'
shutil.copytree(args.input_folder, input_folder_cp)


args.input_folder = input_folder_cp
all_input_paths, all_target_paths = mpi_utils.img_paths4rm_training_directory(args)

Nimgs = len(all_input_paths)
shuffled_Nimgs_arr = np.arange(Nimgs)
np.random.shuffle(shuffled_Nimgs_arr)

shuff_input_paths  = all_input_paths[shuffled_Nimgs_arr]
shuff_target_paths = all_target_paths[shuffled_Nimgs_arr]

print(shuff_input_paths)
print('target paths:')
print(shuff_target_paths)

placeholder_fld = 'placeholder'
if not os.path.isdir(placeholder_fld): os.makedirs(placeholder_fld)

if args.nsplit ==1:
	input_final_dest = os.path.join(args.output_folder, 'all_patient', args.input_gen_folder)
	target_final_dest = os.path.join(args.output_folder,'all_patient', args.target_gen_folder)
	if not os.path.isdir(input_final_dest): os.makedirs(input_final_dest)
	if not os.path.isdir(target_final_dest): os.makedirs(target_final_dest)

	for ip in range(len(all_input_paths)):
		# input and target files rename and move
		rename_n_copy_file(shuff_input_paths, ip, input_final_dest, placeholder_fld, args.file_extension)
		rename_n_copy_file(shuff_target_paths, ip, target_final_dest, placeholder_fld, args.file_extension)
else:
	split_shuff_input_path  =np.array_split(shuff_input_paths,  args.nsplit)
	split_shuff_target_path =np.array_split(shuff_target_paths, args.nsplit)

	for iSp in range(args.nsplit):
		input_final_dest = os.path.join(args.output_folder, 'part_' + str(iSp), 'all_patient', args.input_gen_folder)
		target_final_dest = os.path.join(args.output_folder, 'part_' + str(iSp), 'all_patient', args.target_gen_folder)
		if not os.path.isdir(input_final_dest): os.makedirs(input_final_dest)
		if not os.path.isdir(target_final_dest): os.makedirs(target_final_dest)
		
		for ip in range(len(split_shuff_input_path[iSp])):
			rename_n_copy_file(split_shuff_input_path[iSp], ip, input_final_dest, placeholder_fld, args.file_extension)
			rename_n_copy_file(split_shuff_target_path[iSp], ip, target_final_dest, placeholder_fld, args.file_extension)

os.rmdir(placeholder_fld)
shutil.rmtree(input_folder_cp)


