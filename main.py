##########################################################
# @author: pkc 
#
#
import argparse
import sys
#-------------------------------------------------------------------------------------------------------------------------------------------------
# Command line arguments
#--------------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='sFRC analysis between image pairs from DL(or Reg)- & reference-based methods to identify fake artifacts')
parser.add_argument('--input-folder', type=str, required=True,      help='directory name containing images.')
parser.add_argument('--output-folder', type=str, default='results',  help='output folder to save bounding box-based fake labels on \
                                                                     DL/Reg & reference image pairs, and sFRC plots.')
parser.add_argument('--patch-size',    type=str, default='p96',      help="p96 or p64 or 48 or p32 to indicate patch sizes for the sFRC \
                                                                     analysis. Change padding option below for a different patch size.")
parser.add_argument('--random_N',      action="store_true",          help=" performs sfrc calculation on randomly selected 16 \
                                                                     complimentary images from DL/Reg - Reference folders. \
                                                                     For more info refer to in-built options below.")
parser.add_argument('--input-gen-folder', type=str,                  help="folder name containing DL or regularization method-based outputs.")
parser.add_argument('--target-gen-folder', type=str,                 help="folder name containing reference method-based outputs.")
parser.add_argument('--img-format', type=str, default='dicom',       help='image format for input and target images. Dicom/raw/tif/png?\
                                                                     To add a new image format read function look inside the function \
                                                                     partition_read_normalize_n_augment in file mpi_utils.py.')
parser.add_argument('--multi-patients', action='store_true',         help='if there are multiple-subfolders related to different parents.')
parser.add_argument('--remove-edge-padding', action='store_true',    help='remove patches at the edges of images when mod(img size, patch size) != 0.')
parser.add_argument('--apply-hann', action='store_true',             help='apply hanning filter before the frc calculation.')
parser.add_argument('--frc-threshold', type=str, default='0.5',      help='frc threshold to determine correlation cut-off between the 2 methods. \
                                                                     This patch-based FRC analysis is better suited with a constant threshold such as \
                                                                     0.5, 0.75. Other common options include half-bit, all, one-bit. To add new threshold,\
                                                                     look inside function FRC in the file frc_utils.py.')
parser.add_argument('--inscribed-rings', action='store_true',        help='max frequency at which correlation is calculated is img (or patch) length/2. \
                                                                     if false then frc will be calculated upto the corner of the image (or patch).')
parser.add_argument('--anaRing', action='store_true',                help='perimeter of circle-based calculation to determine data points in each ring. \
                                                                     Otherwise, no. of pixels in each ring used to determine data points in each ring.')
parser.add_argument('--rNx', required=False, type=int, default=None, help="image x-size for raw image as input.")
parser.add_argument('--rNy', required=False, type=int, default=None, help="image y-size for raw image as input. Default is same dim as rNx ")
parser.add_argument('--in-dtype', type=str, required=True,            help="data type of input images. It is needed for .raw format imgs.\
                                                                      It is also needed to set the maximum intensity value for air thresholding \
                                                                      and windowing of patches when saving bounding box-based outputs.")
parser.add_argument('--save-patched-subplots', action='store_true',  help='if you want to save patches with the bounding box and FRC plot results.')
parser.add_argument('--apply-bm3d', action='store_true',             help='apply image-based mild bm3d smoothing before the frc calculation. \
                                                                     It decreases the chance of quick FRC drop. which means it increases the chance of\
                                                                     missing fake artifacts. But it has advantage of increasing PPV.')
parser.add_argument('--mtf-space', action='store_true',              help='x-axis for FRC is in the mtf space. Uses the dx info. \
                                                                     Use this option only if you have info on dx for your acquisition. \
                                                                     Otherwise, do not use this option. When this option is not used x-axis \
                                                                     for FRC has unit pixel(^-1).')
parser.add_argument('--dx', type=float, default=0.48,                help='xy plane pixel spacing. Default value is set from the LDGC dataset and has \
                                                                     the unit mm. ')
parser.add_argument('--ht', type=float, default=0.30,                help='patches whose x-coordinates (corresponding to the points when their FRC curves \
                                                                     intersect with the frc-threshold) fall below this ht threshold will be labeled as fake ROIs.')
parser.add_argument('--windowing', type=str, default='soft',         help='windowing used when generating the patched subplots.\
                                                                     options include soft, lung, bone, unity and none. \
                                                                     Setting appropriate viewing window is very important in zeroing anomalies between a DL method- \
                                                                     and reference method-based outputs. For a sanity check, you may choose to confirm the marked \
                                                                     ROIs generated from this implementation by using software like ImageJ under different type of  \
                                                                     windowing.')
parser.add_argument('--remove-ref-noise', action='store_true',       help='applies a gentle bilateral filtering to reference images.')
parser.add_argument('--img-y-padding', action='store_true',          help='pads y-dim with zeros with pad_width=(rNx-rNy).\
                                                                     It is useful when analyzing coronal-slices.')


if __name__ == '__main__':

    args = parser.parse_args()
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------
    # in-built additional options
    #-------------------------------------------------------------------------------------------------------------------------------------------------------
    args.scale                      = 1 # sfrc-based comparison should be performed between outputs from 2 methods with the same dim
    padding_options                 = {'p512':0, 'p256':0, 'p96':0, 'p64':0, 'p48':0, 'p40':0, 'p36':0, 'p32':0} # patch-size:padding
    args.lr_padding                 = padding_options[args.patch_size]
    patch_option                    = { 'p512':[512 + args.lr_padding, 512 + args.lr_padding],'p256':[256 + args.lr_padding, 256 + args.lr_padding], \
                                        'p96':[96 + args.lr_padding, 96 + args.lr_padding], 'p64':[64 + args.lr_padding, 64 + args.lr_padding], \
                                        'p48':[48 + args.lr_padding, 48 + args.lr_padding], 'p40':[40 + args.lr_padding, 40 + args.lr_padding], \
                                        'p36':[36 + args.lr_padding, 36 + args.lr_padding], 'p32':[32 + args.lr_padding, 32 + args.lr_padding]}
    
    args.input_size, args.label_size= patch_option[args.patch_size]
    args.lr_stride                  = int(args.input_size - args.lr_padding)
    args.channel                    = 1 # code is developed and tested against images with 1 channel
    
    # the array type below is the precision type that MPI performs its operations after the read of data
    # and up until saving the h5 patches. There is not much flexibility in changing the data type for 
    # the MPI operations. Have a look at https://pages.tacc.utexas.edu/~eijkhout/pcse/html/mpi-data.html#Python
    args.dtype                      = 'float32' 
    
    # max value of images (i.e., test set which includes reference and DL/Regularization method-based images). 
    # this is only used when 'unity' is set as the windowing option
    # look inside the function dict_plot_of_2d_arr in plot_func.py
    # it is also used to normalize uint8 images. So that we can apply sfrc air thresholding
    # look into partition_read_normalize_n_augment function in mpi_utils under image format type png or tif
    if args.in_dtype == 'uint8': 
        # for MRI dataset
        args.img_list_max_val= 255.0 
    elif args.in_dtype == 'uint16': 
        # for CT dataset
        args.img_list_max_val= 2686
    else:
        args.img_list_max_val= 1.0
        print('*************************************************************************************')
        print('WARNING. Ensure that the max value of images is accurately set in the file main.py.')
        print('*************************************************************************************')
    
    # default for y-dim is same as x-dim unless specified as command line argument using rNy
    if args.img_format=='raw':
        if args.rNy == None:
            args.rNy=args.rNx
    #-------------------------------------------------------------------------------------------------------------------------
    # following options are inherited from mpi patching code 
    # but are hard-coded to the parameters below for this sFRC implementation
    #-------------------------------------------------------------------------------------------------------------------------
    
    # option below dictates DL/reg & ref image pair normalization. 
    # It is set to yield un-normalized images from the 2 methods.
    # however, after patch-pair is formed they are normalized before the frc comparison. 
    # This way it will be easier to save patched plots at different window level. 
    args.normalization_type              = None
    
    # the below air threshold is for patching from old code. 
    # and not for patches for the sFRC calculation 
    # air_threshold is hard coded to apply for sFRC. 
    # look in the function patchwise_sfrc in utils.py.
    # however threshold values for uint8 type (or ) may need to be re-tuned. 
    args.air_threshold                  = False
    
    # sFRC is not calculated on any augmented patches. 
    # so turning off patch augmentation options.
    args.ds_augment                     = False
    args.rot_augment                    = False
    args.dose_blend                     = False
    args.blurr_n_noise                  = False
    
    # perform sfrc calculation on randomly selected 16 pairs of input and target images
    # rather than going through all the images for a quick validation of this sfrc-based analysis. 
    if args.random_N: args.N_rand_imgs  = 16 
    
    # this codes was developed over the mpi-based patch generation code that extract
    # patches from a given full image. 
    # hence, mpirun needs to be turned on. 
    args.mpi_run                        = True
    
    if(args.mpi_run==True): from sfrc_in_mpi import sfrc_in_mpi; sfrc_in_mpi(args)	
	