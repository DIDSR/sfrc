#########################################################################
# @author: pkc 
#
# utils.py includes generic signal processing and mathematical operations
# based functions.
#

import os 
import glob
import numpy as np 
import random 
import cv2
import sys

import plot_func as pf
from random import randrange
import frc_utils
import io_func

# Ignoring RuntimeWarnings  
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def add_white_noise(arr, mu, sigma, factor, size):
    """
    add gaussian noise to input numpy array
    
    inputs
    ------
    arr   : input array 
    sigma : standard deviation of the distribution 
    mu    : mean of the distribution
    factor: scalar pre-factor applied to gaussian noise before adding to the input array
    size  : shape of input array
    
    output
    ------
    numpy array with added gaussian noise
    """
    noisy_arr = arr + factor * np.random.normal(loc = mu, scale = sigma, size = size) 
    return noisy_arr

def add_rnl_white(rnl, b, mu, sigma):
    """ 
    add relative white noise such that
    norm2(input array)/norm2(added noise) = relative ratio
    
    input
    -----
    rnl  : relative ratio
    b    : input array
    mu   : mean of the initialized random noise
    sigma: standard deviation initialized random noise
    
    output
    ------
    array with added relative noise
    note that the mu and sigma of the added noise will
    be different than that provided as input because the 
    objective here to add relative noise.
    """
    h, w = b.shape
    randn =  np.random.normal(loc = mu, scale = sigma, size = (h,w))
    e = randn/np.linalg.norm(randn, ord = 2)
    e = rnl*np.linalg.norm(b)*e;
    return(b + e)

def normalize_data_ab(a, b, data):
    """
    Normalizes the input data to a new range (b-a).
    
    inputs
    -----
    data: numpy array to be normalized
    a   : min value of the normalized output
    b   : max value of the normalized output
    
    yields
    -------
    normalized data
    """
    min_x = min(data.ravel())
    max_x = max(data.ravel())  
    range_x = max_x - min_x 
    return((b-a)*((data-min_x)/range_x)+a)

def normalize_data_ab_cd(a, b, c, d, data):
    """
    Normalizes the input data with a range (d-c) to a new range (b-a).
    
    inputs
    -----
    data: numpy array to be normalized
    c   : min value of input array
    d   : max value of input array
    a   : min value of the normalized output
    b   : max value of the normalized output
    
    returns
    -------
    normalized data
    """
    min_x = c
    max_x = d  
    range_x = max_x - min_x 
    return((b-a)*((data-min_x)/range_x)+a)

def modcrop(image, scale=3):
    """ 
    To ensure that transition between HR to LR 
    and vice-versa is divisible by the scaling 
    factor without any remainder the input image
    is cropped relative the scaling factor
    
    input
    -----
    image: input 2d or 3d array to be croped
    scale: super-resolution scaling factor
    
    output
    -----
    cropped array such that mod((h, w) of output, scale)=0
    """
    if len(image.shape) == 3:
      h, w, _ = image.shape
      h = h - np.mod(h, scale)
      w = w - np.mod(w, scale)
      image = image[0:h, 0:w, :]
    else:
      h, w = image.shape
      h = h - np.mod(h, scale)
      w = w - np.mod(w, scale)
      image = image[0:h, 0:w]
    return image

def get_sorted_random_ind(foldername, Nimgs):
    """
    Outputs N random numbers between 0 and the total number 
    of images in path foldername.
    Suppose foldername has 20 images and Nimgs is set has 8 then
    this functions outputs a 1d array with 8 random numbers between
    0 and 20. 
    
    input
    -----
    foldername (str): path with images to determine total number of images
    Nimg (int): total number of random numbers to output
    
    output
    -------
    random Nimgs sized 1d array.
    """
    data_dir = os.path.join(os.getcwd(), foldername)
    images = sorted(glob.glob(os.path.join(foldername, "*.*")))
    randind = sorted(random.sample(range(len(images)), int(Nimgs)))
    return(randind)

def getimages4rmdir(foldername, randN=None):
    """
    Outputs filename array (with paths) of files 
    in the input folder string 
    ...
    
    input
    -----
    foldername (str): path where images are stored
    randN (optional): int array on indices (if provided) will return
                      image paths corresponding to those indices only.
    
    output
    ------
    array of filepaths.
    """
    # sorted is true by default to remain consistent 
    data_dir = os.path.join(os.getcwd(), foldername)
    images = sorted(glob.glob(os.path.join(data_dir, "*.*")))

    if (randN !=None):
      images = np.array(images)
      images = list(images[randN]) 
    return images

# cv2 intakes only float32 data types
# neg values dues to dose augmentation is accounted
# rotation or ds augmentation does not yield neg values
def downsample_4r_augmentation(initial_image):
    """
    cv2's area-based downsampled interpolation of the input array
    ...
    
    input
    -----
    initialimage (float32): input array
    
    output
    ------
    appended arrays with downsampled 2 matrices by factor 0.8 and 0.6.
    """
    #aug_ds_facs = np.asarray([0.9, 0.8, 0.7, 0.6])
    # cv2 only works for float type 32 
    aug_ds_facs = np.asarray([0.8, 0.6])
    aug_ds_input = []
    h, w = initial_image.shape
    #print(initial_image.shape)
    aug_ds_input.append(initial_image)
    for i in range(len(aug_ds_facs)):
        scale = aug_ds_facs[i]
        aug_label = cv2.resize(initial_image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        #sys.exit()
        aug_ds_input.append(aug_label)
    return(aug_ds_input)

def bn_seed(Nlen, bratio, nratio):

  """ 
  Returns a three valued (with 1s, 2s, & 3s) array with 
  length (Nlen).the distribution of the three values is 
  randomly assigned in the 1d-based output of the function. 
  bratio*Nlen of the array will be assigned as 1 and the associated images
  will be blurred. Likewise nratio*Nlen will be assigned as 2
  and the corresponding images will be noised. The remaining indices will
  be assigned a value of 3
  
  input
  ------
  Nlen (int)    : length of the 1d array
  bratio (float): 0 to 1 valued float
  nratio (float): 0 to 1 valued float
  
  output
  ------
  1d array which consists of 1, 2, and 3 as its unique values
  """

  blist           = random.sample(range(Nlen), int(bratio*Nlen))
  rlist           = list(set(range(Nlen))-set(blist))
  nlist           = random.sample(rlist, int(nratio*Nlen))
  rlist           = list(set(rlist) - set(nlist))
  bn_seed         = np.zeros(Nlen)
  bn_seed[blist]  = 1
  bn_seed[nlist]  = 2
  bn_seed[rlist]  = 3
  return(bn_seed)

def min_max_4rm_img_name_arr (img_nm_arr, img_type, dtype, rNx=256):
    """
    reads images filename list provided as array.
    Then outputs min and max values of the images
    
    input
    -----
    img_nm_arr : array of filenames
    img_type   : string with options such as 'dicom', 'raw', 'png', 'tif'
    dtype      : data type of input files such as 'uint8', 'uint16'
    rNx        : image size (used only for raw files)
    
    output
    ------
    min and max values
    """
    dir_max, dir_min = 0.0, 0.0
    for i in range(len(img_nm_arr)):
        if img_type=='dicom':
            img = io_func.pydicom_imread(img_nm_arr[i])
        elif img_type=='raw':
            img = io_func.raw_imread(img_nm_arr[i], (rNx, rNx), dtype=dtype)
        else:
            img = io_func.imageio_imread(img_nm_arr[i])

        img_max, img_min = np.max(img), np.min(img)
        if (img_max > dir_max): dir_max = img_max
        if (img_min < dir_min): dir_min = img_min
        #print(img_min, img_max)
    return(dir_min, dir_max)

def interpolation_hr(norm_lr, scale):
  """cv2-based upsampling of input array by the parameter scale"""
  h, w = norm_lr.shape
  #norm_hr = resize(norm_lr, (h *scale, w*scale), anti_aliasing=True)
  norm_hr = cv2.resize(norm_lr, (w*scale, h*scale), interpolation=cv2.INTER_AREA)
  return norm_hr 

def interpolation_lr(hr_image, scale):
  """cv2-based downsampling of input array by the parameter scale"""
  h, w = hr_image.shape
  lr = cv2.resize(hr_image, (int(w*(1/scale)), int(h*(1/scale))), interpolation=cv2.INTER_AREA)
  return(lr)

def add_blurr_n_noise(input_, seed):
  """
  Blur (when seed=1), or add gaussian noise 
  (when seed=2), or leave the input array as 
  is (when seed!=(1, 2)).
  """
  if (seed == 1):
    # adding blurr
    corrupted    = cv2.blur(input_, (3,3))
 
  elif (seed ==2):
    # adding noise
    h, w         = input_.shape
    mu           = 0.0
    sigma        = 8.0
    # noise        = np.random.normal(mu, sigma, (h, w))
    # corrupted    = input_ + noise
    corrupted    = add_rnl_white(0.01, input_, mu, sigma)
    corrupted    = normalize_data_ab(np.min(input_), np.max(input_), corrupted)

  else:
    # doing nothing 
    corrupted    = input_
  return(corrupted)

def overlap_based_sub_images(config, input_, label_):
  """
  converts full sized image pairs (input and target) into 
  complimentary patches
  
  inputs:
  -------
  config: parser.parse_ags()
          command line arguments and in-built options set in main.py file.
          The options used in this parameters include patch size, strides 
          between the patches, scaling factor for the patches and padding
          of the patches
  input_: degraded full-sized image used in model training or testing
  label_: reference full-sized image used in model training or testing
  
  output
  ------
  two 3D arrays as complimentary patches from input and target image pairs
  """
  image_size, label_size, lr_stride, scale, lr_padding = config.input_size, config.label_size, config.lr_stride, config.scale, config.lr_padding 
  hr_padding = lr_padding*scale

  ih, iw = input_.shape
  lh, lw = label_.shape
  sub_input_of_one_input, sub_label_of_one_label = [], []
  for x in range(0, ih, lr_stride):
      if(x==0):
          L_xs  = x
          L_xe  = x + (image_size )
      
          H_xs  = x
          H_xe  = x + (label_size )
      else:    
          L_xs  = x - lr_padding
          L_xe  = (x - lr_padding) + image_size
      
          H_xs  = x*scale -hr_padding
          H_xe  = (x*scale -hr_padding) + label_size
      
      if (L_xe >=ih):
          L_xe = ih-1
      if (H_xe >=lh):
          H_xe =lh -1
          
      # print("Lr: xs, xe: ", L_xs, L_xe, "\t\thr: xs, xe: ", H_xs,  H_xe)
      for y in range(0, iw, lr_stride):
      
          if(y==0):
              L_ys = y
              L_ye = y + (image_size )
              
              H_ys = y
              H_ye = y + (label_size )
          else:    
              L_ys = y - lr_padding
              L_ye = (y - lr_padding) + image_size 
          
              H_ys = y*scale - hr_padding
              H_ye = (y*scale - hr_padding) + label_size        
          
          if (L_ye >=iw):
              L_ye = iw - 1
          if (H_ye >=lw):
              H_ye = lw -1
              
          #print("\tLr: ys, ye: ", L_ys,  L_ye, "\t\tHr: ys, ye: ",  H_ys,  H_ye)
          sub_input = input_[L_xs : L_xe, L_ys : L_ye]
          sub_label = label_[H_xs : H_xe, H_ys : H_ye]

          
          # the default is edge padding of patches at the edges of the image with 0
          # however, if the remove_edge_padding is passed as an arguement to the program
          # these patches at the edges are not considered in calculation
          if config.remove_edge_padding: 
              if (sub_input.shape!=(image_size, image_size)):
                break
          else:
              if (sub_input.shape!=(image_size, image_size)):
                sih, siw = sub_input.shape
                sub_input = np.pad(sub_input,((0, image_size-sih),(0,image_size-siw)) , 'constant')
                #print('shape not eq', sub_input.shape)
          
              if (sub_label.shape!=(label_size, label_size)):
                slh, slw = sub_label.shape
                sub_label = np.pad(sub_label,((0, label_size-slh),(0,label_size-slw)) , 'constant')
                #print('label shape not eq', sub_label.shape)

          sub_input = sub_input.reshape([image_size, image_size, 1])
          sub_label = sub_label.reshape([label_size, label_size, 1])
          
          sub_input_of_one_input.append(sub_input)
          sub_label_of_one_label.append(sub_label)

  return(np.asarray(sub_input_of_one_input), np.asarray(sub_label_of_one_label))

def rotation_based_augmentation(args, sub_input, sub_label):
  """
  This function is not used in the sFRC calculation.
  #-----------------------------------
  # ROTATION BASED DATA AUGMENTATION       
  #----------------------------------
  # in addition to the regular upright positioned
  # images, we also include the same image
  # rotated by 90,180 , 270 degs or flippped 
  # LR or UD in our training
  # instances
  # sub_input is of shape [chunk_size, input_size, input_size, 1]
  """
  rotated_all_inputs = np.empty([0, args.input_size, args.input_size, 1])
  rotated_all_labels = np.empty([0, args.label_size, args.label_size, 1])
  
  #print("inside rotation subroutine", sub_input.shape, sub_label.shape)
  #sys.exit()
  for exp in range(3):
    if exp==0:
        add_rot_input = sub_input
        add_rot_label = sub_label
    elif exp==1:
        # rotates given patch either by 90 deg or 180 deg or 270 deg
        k = randrange(3)+1
        add_rot_input = np.rot90(sub_input, k, (1,2))
        add_rot_label = np.rot90(sub_label, k, (1,2))
    elif exp==2:
        # flips either the Patch either LR or UP
        k = randrange(2)
        add_rot_input = sub_input[:,::-1] if k==0 else sub_input[:,:,::-1]
        add_rot_label = sub_label[:,::-1] if k==0 else sub_label[:,:,::-1]      

    rotated_all_inputs =  np.append(rotated_all_inputs, add_rot_input, axis=0)
    rotated_all_labels =  np.append(rotated_all_labels, add_rot_label, axis=0)

  return(rotated_all_inputs, rotated_all_labels)

def dose_blending_augmentation(args, sub_input, sub_label, blend_factor):
  """
  This function is not used in sFRC-based calculations.
  # dose blending can cause certain patches to exhibit negative values
  # Therefore minimum of these neg values should be added as (-min(patch))
  # to the corresponding input and label patches
  """
  blended_all_inputs = np.empty([0, args.input_size, args.input_size, 1])
  blended_all_labels = np.empty([0, args.label_size, args.label_size, 1])

  for exp in range(2):
    if exp==0:
        add_blend_input = sub_input
        add_blend_label = sub_label
    elif exp==1:
        add_blend_label = sub_label
        add_blend_input = sub_label + blend_factor*(sub_input-sub_label)
        # blending might include neg values for certain patches
        # so adding the neg of those neg values for each of these patches in input as well as label
        if (np.min(add_blend_input) < 0.0):
          neg_ind = np.where(add_blend_input < 0.0)
          neg_ind_unq_ax0 = np.unique(neg_ind[0])
          # get negative values for each patch w.r.t axis 0
          neg_vals_arr = add_blend_input[neg_ind_unq_ax0, :, :, :]
          min_neg_vals = np.squeeze(np.min(np.min(neg_vals_arr, axis=1), axis=1))
          for i in range(len(neg_ind_unq_ax0)):
            add_blend_input[neg_ind_unq_ax0[i]] = add_blend_input[neg_ind_unq_ax0[i]] + (-min_neg_vals[i])
            add_blend_label[neg_ind_unq_ax0[i]] = add_blend_label[neg_ind_unq_ax0[i]] + (-min_neg_vals[i])
          #print("inside neg rotation subroutine", add_blend_input.shape, add_blend_label.shape, (np.min(add_blend_input)), np.min(add_blend_label))
          #sys.exit()
    
    blended_all_inputs = np.append(blended_all_inputs, add_blend_input, axis=0)
    blended_all_labels = np.append(blended_all_labels, add_blend_label, axis=0)
  return(blended_all_inputs, blended_all_labels)
    
def img_pair_normalization(input_image, target_image, normalization_type=None):
  """
  Normalizing the input (LD) and target (HD) image-pairs as per the normalization type
  set as input string. 
  
  input
  -------
  input_image        : input  as a float array
  target_image       : target as a float array 
  normalization_type : string options: 
      unity_independent  : each of the image-pairs are seperately normalized to the range [0,1]
      max_val_independent: both LD-HD pair are independently normalized to from (min_val, max_val) 
                           to (min_val/max_val, min_val/max_val)
      unity_wrt_ld       : both LD-HD pairs are normalized to from (min_val, max_val) to (0, 1) 
                           with range (LD_max_val - LD_min_val)
      max_val_wrt_ld     : both LD-HD pair is normalized to from (min_val, max_val) to 
                           (min_val/LD_max_val, min_val/LD_max_val) 
      std_independent    : LD-HD pair is independently stardarized based on their respective values.
      dicom_unity        : LD-HD pairs are normalized between (0,1) considering they exhibit 
                           (0, 2^16) as their initial range
      dicom_std          : LD-HD pairs are standardrized  considering they exhibit 
                           (0, 2^16) initial value range.
      set_max_val        : LD-HD pairs are normalized by 4094
  
  output
  --------
  normalized input-target array pairs             
  """
  if normalization_type == 'unity_independent':
    # both LD-HD pair are independently normalized to from (min_val, max_val) to (0, 1)
    out_input_image  = normalize_data_ab(0.0, 1.0, input_image)
    out_target_image = normalize_data_ab(0.0, 1.0, target_image)
    
  elif normalization_type == 'max_val_independent':
    # both LD-HD pair are independently normalized to from (min_val, max_val) to (min_val/max_val, min_val/max_val)
    if np.min(input_image)<0: input_image += (-np.min(input_image))
    if np.min(target_image)<0: target_image += (-np.min(target_image))
    out_input_image  =input_image/np.max(input_image)
    out_target_image =target_image/np.max(target_image)

  elif normalization_type == 'unity_wrt_ld':
    # both LD-HD pair are normalized to from (min_val, max_val) to (0, 1) with range (LD_max_val - LD_min_val)
    out_target_image = normalize_data_ab_cd(0.0, 1.0, np.min(input_image), np.max(input_image), target_image)
    out_input_image  = normalize_data_ab(0.0, 1.0, input_image)

  elif normalization_type == 'max_val_wrt_ld':
    # both LD-HD pair is normalized to from (min_val, max_val) to (min_val/LD_max_val, min_val/LD_max_val) 
    if np.min(input_image)<0: input_image += (-np.min(input_image))
    if np.min(target_image)<0: target_image += (-np.min(target_image))
    out_target_image =target_image/np.max(input_image)
    out_input_image =input_image/np.max(input_image)
  
  elif normalization_type == 'std_independent':
    # LD-HD pair is independently stardarized based on their respective values
    out_input_image = (input_image - np.mean(input_image))/(np.max(input_image)-np.min(input_image))
    out_target_image = (target_image - np.mean(target_image))/(np.max(target_image)-np.min(target_image))

  elif normalization_type == 'std_wrt_ld':
    # LD-HD pair is jointly stardarized based on LD values 
    out_target_image = (target_image - np.mean(input_image))/(np.max(input_image)-np.min(input_image))
    out_input_image = (input_image - np.mean(input_image))/(np.max(input_image)-np.min(input_image))
  
  # for dicom unity  as well as dicom_std everything has been scaled between (0, 2^16) hence there
  # is only one option for both and not the independent and wrt_ld types 
  elif normalization_type == 'dicom_unity':
    # all LD-HD pair are normalized between (0,1) while considering they exhibit (0, 2^16) initial value range
    out_input_image = normalize_data_ab_cd(0.0, 1.0, 0.0, 2.0**12, input_image)
    out_target_image = normalize_data_ab_cd(0.0, 1.0, 0.0, 2.0**12, target_image)
  
  elif normalization_type == 'dicom_std':
    # all LD-HD pair are standarized while considering they exhibit (0, 2^16) initial value range
    out_input_image = (input_image - np.mean(input_image))/(2.0**12)
    out_target_image = (target_image - np.mean(target_image))/(2.0**12)

  elif normalization_type =='set_max_val':
    out_input_image = input_image/4094.0
    out_target_image = target_image/4094.0
 
  else:
    out_input_image = input_image
    out_target_image = target_image

  return(out_input_image, out_target_image)

def air_thresholding(args, sub_input, sub_label, sub_label_un, return_ind=False):
  """
  air threshold patches in which 
  (10% of (total no. of its pixel) is below pix thres value)
  & (the patch average is below mean threshold),
  i.e., trash them out for not having
  enough information for either training Deep learning models
  or correlation-based comparison.
  
  It might be advisable to change mean thres value depending on 
  air values (or minimum value). 
  i.e. sometimes training data is not scaled and air is at its original
  HU unit i.e., -1024 HU. Also, sometimes, pixel value of 25 or in int16 
  image with pixel value of 200 may correspond to air.
  default CT values. 
  pix_thresh   = 200 
  count_thresh = int(0.1* (args.input_size**2))
  mean_thresh  = 100 
  
  This air thresholding was developed using LDGC training data where the 
  maximum HU intensity was 2686. Hence, for other modality-based data with a 
  different range in its intensity values, 
  pix_thresh and mean_thresh will have to be accordingly adjusted
  """
  pix_thresh   = args.img_list_max_val*200/2686 #*1.5 -> mri
  count_thresh = int(0.1* (args.input_size**2))
  mean_thresh  = args.img_list_max_val*150/2686 #*2.5 -> mri
  threshold_pass_ind = []
  # print('inside air thresholding')
  # pf.dict_plot_of_2d_arr(args, 7, 7, np.zeros((7,7)), sub_label[:, :, :, 0], save_plot=False, disp_plot=True, output_path='', plt_title=str(0))
  for i in range(len(sub_label_un)):
    patch_orig = sub_label_un[i]
    num_pix_above_thresh = np.sum(patch_orig > pix_thresh)
    
    if (num_pix_above_thresh > count_thresh) & (patch_orig.mean() > mean_thresh):
      threshold_pass_ind.append(i)
    
    #print('i=', i, 'num_pix_above_thresh=', num_pix_above_thresh, 'count_thresh=', count_thresh, 'patch_orig=', patch_orig.mean(), 'mean_thresh=', mean_thresh)

  threshold_pass_ind=np.asarray(threshold_pass_ind).astype('int')

  if return_ind:
    return(threshold_pass_ind)
  else:
    return(sub_input[threshold_pass_ind], sub_label[threshold_pass_ind])

def patchwise_sfrc(args, sub_input, sub_label, ch_ind, pid, chunk_sz, img_name, output_folder, output_patched_folder):
  """
  Extracts complimentary patches from a given image pairs (input-target), performs sFRC and outputs
  fakes as red bounded box.
  
  input
  -------
  args                  : parser.parse_ags() from the command line
  ch_ind                : chunk index in the range(total number of input images pairs/No. of processors)
  pid                   : rank number in from mpi
  chunk_sz              : total number of input images pairs/No. of processors
  sub_input             : patches corresponding to an image from first method for the sFRC calculation
  sub_target            : patches corresponding to an image from second method for sFRC calculation
  img_name              : image string name of either input or target images used to extract image number from the 
                          stack of image-pairs in which you seek to perform the sFRC.
  output_folder         : folder name to store sfrc-based corelation curve for all the patches in a given input-target
                          image pairs
  output_patched_folder : folder name to store sfrc-based fake ROIs indicated using red-bounded box.
                          it includes additional tags in terms of threshold, filtering, MTF vs FRC space used
                          
  output
  -------
  two folders with images stored as
  (1) red-bounded box deemed as fakes in input images
  (2) sFRC line curves on the patches from input-target image pairs
  """
  if sub_input.shape != sub_label.shape:
    print('ERROR! Mismatch between the dim of patches from the 2 methods.')
    print('method 1 patch size = ', sub_input.shape, 'method 1 patch size =', sub_label.shape)
    sys.exit()
  #pf.dict_plot_of_2d_arr(6, 6, sub_input[0:36, :, :, 0], save_plot=False, disp_plot=True, output_path='', plt_title='patches of img: '+str(pid*chunk_sz+ch_ind))
  
  frc_len = int(args.label_size/2)
  each_rank_pw_stacked_frc = np.empty([0, frc_len], dtype=args.dtype)
  # -----------------------------------------------------------------------------------------------------
  # sequentially through patches within a given image-pair (i.e. a pair from 2 methods used to calculate FRC)
  # -----------------------------------------------------------------------------------------------------
  img_str = img_name
  img_str = img_str.split('/')[-1]
  img_tp  = img_str.split('.')[-2] #img type ref vs method with string based on the image name... eg gt_0
  img_no  = pid*chunk_sz+ch_ind
  for i in range(len(sub_label)):
    _gt_patch = np.squeeze(sub_label[i])
    _me_patch = np.squeeze(sub_input[i])

    if args.img_y_padding:
      if(np.all(_gt_patch==0.0)):
        pass # leave _gt and _me as zero matrix
      else:
        _gt_patch = normalize_data_ab(0.0, 1.0, _gt_patch)
        _me_patch = normalize_data_ab(0.0, 1.0, _me_patch)
    else:
      _gt_patch = normalize_data_ab(0.0, 1.0, _gt_patch)
      _me_patch = normalize_data_ab(0.0, 1.0, _me_patch)

    if args.apply_hann:
      _gt_patch = frc_utils.apply_hanning_2d(_gt_patch)
      _me_patch = frc_utils.apply_hanning_2d(_me_patch)

    # patch wise FRC comparision
    # FRC thresholding such as one-bit or half-bit the default info_split of true may need to be changed to false
    # for contast thresholds such as 0.5, it does not matter whether info_split is turned on or off
    xc, corr_avg, xt, thres_val = frc_utils.FRC(_gt_patch, _me_patch, thresholding=args.frc_threshold, inscribed_rings=args.inscribed_rings, analytical_arc_based=args.anaRing)
    each_rank_pw_stacked_frc    = np.append(each_rank_pw_stacked_frc, corr_avg.reshape(1, frc_len), axis=0)

  # change the x coordinate if mtf_space is true
  if args.mtf_space:
    xc = np.linspace(0, 1.0/(2*args.dx), int(args.label_size/2))
    xt = xc

  air_threshold_pass_ind = air_thresholding(args, sub_input, sub_label, sub_label, return_ind=True)

  # initialization as false to all patches
  im_wise_hallu_bool_arr1d = np.zeros((len(sub_label))) 
  im_wise_hallu_bool_arr1d[air_threshold_pass_ind] = 1.0 # candidate hallucination patches are the ones that pass the air thresholding

  # one frc plot (or one plot with all subplots of patches)
  frc_plot_name = output_folder + 'sfrc_'+ str(img_no) + '.png'
  if len(sub_label)==1:
    pf.plot_n_save_img(threshold=args.frc_threshold, fx_coord=xc, frc_val=each_rank_pw_stacked_frc[0], tx_coord=xt, thres_val=thres_val, output_img_name=frc_plot_name, save_img=True, display_img=False, plt_title=None, mtf_space=args.mtf_space)
  else:
    final_hallu_bool_mat, per_ip_fk = pf.dict_plot_of_patched_frc(bool_hallu_1darr=im_wise_hallu_bool_arr1d, args=args, fx_coord=xc, stacked_frc=each_rank_pw_stacked_frc, tx_coord=xt, thres_val=thres_val, output_img_name=frc_plot_name, save_img=True, display_img=False, plt_title=None, mtf_space=args.mtf_space)

  print('rank=', pid, '|| chunk_ind=', ch_ind, '|| img no=', pid*chunk_sz+ch_ind, '|| target_size=', sub_label.shape, '|| stacked_frc shape=', each_rank_pw_stacked_frc.shape, '|| No. of airT passed patches =', len(air_threshold_pass_ind), '|| No. of fake patches=', per_ip_fk)
  if args.save_patched_subplots:
    input_patched_im_name = output_patched_folder + 'method/me_'+ str(img_no) + '.png'
    target_patched_im_name= output_patched_folder + 'ref/gt_'+ str(img_no) + '.png'
    Npatches              = len(sub_label)
    Nrows                 = int(np.ceil(np.sqrt(Npatches)))
    Ncols                 = Nrows
    pf.dict_plot_of_2d_arr(args, Nrows, Ncols, final_hallu_bool_mat, sub_input[:, :, :, 0], save_plot=True, disp_plot=False, output_path=input_patched_im_name, plt_title=str(pid))
    pf.dict_plot_of_2d_arr(args, Nrows, Ncols, final_hallu_bool_mat, sub_label[:, :, :, 0], save_plot=True, disp_plot=False, output_path=target_patched_im_name, plt_title=str(pid))
  return(per_ip_fk)

