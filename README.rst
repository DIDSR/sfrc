sFRC for detecting fakes in AI-assisted medical image recovery
=========================================================================
This implementation performs Fourier Ring Correlation (FRC)-based analysis over small patches and concomitantly (s)canning
across Deep learning(DL) or regularization(Reg)-based outputs and their reference counterparts to identify fakes.


Usage
-----

.. code-block:: bash

    $ main.py [-h] --input-folder INPUT_FOLDER [--output-folder OUTPUT_FOLDER] [--patch-size PATCH_SIZE] [--random_N]
              [--input-gen-folder INPUT_GEN_FOLDER] [--target-gen-folder TARGET_GEN_FOLDER] [--img-format IMG_FORMAT] 
              [--multi-patients] [--remove-edge-padding] [--apply-hann] [--frc-threshold FRC_THRESHOLD] [--inscribed-rings] 
              [--anaRing] [--rNx RNX] [--rNy RNY] --in-dtype IN_DTYPE [--save-patched-subplots] [--apply-bm3d] [--mtf-space]
              [--dx DX] [--ht HT] [--windowing WINDOWING] [--remove-ref-noise] [--img-y-padding]

    sFRC analysis between image pairs from DL(or Reg)- & reference-based methods to identify fake artifacts
    arguments:
    -h, --help            show this help message and exit
    --input-folder        directory name containing images.
    --output-folder       output folder to save bounding box-based fake labels on DL/Reg & reference image pairs, and sFRC plots.
    --patch-size          p96 or p64 or 48 or p32 to indicate patch sizes for the sFRC analysis. Change padding option below for a
                          different patch size.
    --random_N            performs sfrc calculation on randomly selected 16 complimentary images from DL/Reg - Reference folders.
                          For more info refer to in-built options in main.py.
    --input-gen-folder    folder name containing DL or regularization method-based outputs.
    --target-gen-folder   folder name containing reference method-based outputs.
    --img-format          image format for input and target images. Dicom/raw/tif/png? To add a new image format read function look 
                          inside the function partition_read_normalize_n_augment in file mpi_utils.py.
    --multi-patients      if there are multiple-subfolders related to different parents.
    --remove-edge-padding remove patches at the edges of images when mod(img size, patch size) != 0.
    --apply-hann          apply hanning filter before the frc calculation
    --frc-threshold       frc threshold to determine correlation cut-off between the 2 methods. This patch-based FRC analysis
                          is better suited with a constant threshold such as 0.5, 0.75. Other common options include half-bit, all,
                          one-bit. To add new threshold, look inside function FRC in the file frc_utils.py
    --inscribed-rings     max frequency at which correlation is calculated is img (or patch) length/2. if false then frc will be
                          calculated upto the corner of the image (or patch).
    --anaRing             perimeter of circle based calculation to determine data points in each ring. Otherwise no. of pixels in
                          each ring used to determine data points in each ring.
    --rNx RNX             image x-size for raw image as input.
    --rNy RNY             image y-size for raw image as input. Default is same dim as rNx
    --in-dtype            data type of input images. It is needed for .raw
                          format imgs. It is also needed to set the maximum intensity value for air thresholding and windowing of
                          patches when saving bounding box-based outputs.
    --save-patched-subplots
                          if you want to save patches with the bounding box and FRC plot results.
    --apply-bm3d          apply image-based mild bm3d smoothing before the frc calculation. It decreases the chance of quick FRC
                          drop. which means it increases the chance of missing fake artifacts. But it has advantage of increasing PPV.
    --mtf-space           x-axis for FRC is in the mtf space. Uses the dx info. Use this option only if you have info on dx for your
                          acquisition. Otherwise do not use this option. When this option is not used x-axis for FRC has unit pixel(^-1).
    --dx                  xy plane pixel spacing. Default value is set from the LDGC dataset and has the unit mm.
    --ht                  patches whose x-coordinates (corresponding to the points when their FRC curves intersect with the frc-
                          threshold) fall below this ht threshold will be labeled as fake ROIs.
    --windowing           windowing used when generating the patched subplots Options include soft, lung, bone, unity and none.
                          Setting appropriate viewing window is very important in zeroing anomalies between a DL method- and
                          reference method-based outputs. For a sanity check, you may choose to confirm the marked ROIs generated
                          from this implementation by using software like ImageJ under different type of windowing.
    --remove-ref-noise    applies a gentle bilateral filtering to reference images
    --img-y-padding       pads y-dim with zeros with pad_width=(rNx-rNy). Its useful when analyzing coronal-slices


DL/Reg method- and Reference method-based data for sFRC 
----------------------------------------------------------

1. Get SRGAN-based CT upsampled (x4) output
----------------------------------------------------------

Usage::
  cd ctsr
  chmod +x demo_srgan_test.sh 
  ./demo_srgan_test.sh 'sh' 'sel'

'sh' indicates sharp kernel-based test set and 'sel' indicates CT images used as tuning set for sFRC parameters in our paper.
Likewise 'sm'indicates smooth kernel-based test set and '' indicates CT images used as test set for sFRC analysis in our paper.
To apply the trained SRGAN model on all CT images from patient L067 look inside the file ctsr/create_sr_dataset/readme.txt to
get the required LDGC box path and on how to get the downsampled input.
----


2. Get UNet- and PLS-TV-based recovery of subsampled (3x) acquisition
----------------------------------------------------------
All the post-processing codes, data have been sourced from . Other packages such as BART and fastmri are 

Usage::

  python main_3d.py --acceleration_factor 4

edit the path to BART's python wrapper in line 20 in file mrsub/plstv/bart_pls_tv.py
  cd mrsub/unet
  chmod +x run_unet_test.sh
  ./run_unet_tesh.sh
----

3. sFRC analysis on the SRGAN-based outputs
----------------------------------------------------------

Reconstruct dynamic MR images from its undersampled measurements using 
Convolutional Recurrent Neural Networks. This is a pytorch implementation requiring 
Torch 0.4.  

Usage::

  ./demo_sfrc_run.sh 'sh' 'sel' #on sharp kernel-based tuning set

Once you successfully download and preprocess test CT scans of patient L067 used in the paper
  ./demo_sfrc_run 'sh' '' 47 #on sharp test data with 47 set as no. of processors
  ./demo_sfrc_run 'sm' '' 47 #on smooth test data with 47 set as the no. of processors
----

4. sFRC analysis on the UNet-based output
----------------------------------------------------------

Reconstruct dynamic MR images from its undersampled measurements using 
Convolutional Recurrent Neural Networks. This is a pytorch implementation requiring 
Torch 0.4.  

Usage::

  python main_crnn.py --acceleration_factor 4


References 
----------
1. McCollough, Cynthia H., et al. "Low‐dose CT for the detection and classification of metastatic liver lesions: results of the 2016 low dose CT grand challenge." Medical physics 44.10 (2017): e339-e352.

2. Bhadra, Sayantan, et al. "On hallucinations in tomographic image reconstruction." IEEE transactions on medical imaging 40.11 (2021): 3249-3260.

3. `hallucinations-tomo-recon <https://github.com/comp-imaging-sci/hallucinations-tomo-recon>`_.

4. Ledig, Christian, et al. "Photo-realistic single image super-resolution using a generative adversarial network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

5. Sergeev, Alexander, and Mike Del Balso. "Horovod: fast and easy distributed deep learning in TensorFlow." arXiv preprint arXiv:1802.05799 (2018).

6. Uecker, Martin, et al. "The BART toolbox for computational magnetic resonance imaging." Proc Intl Soc Magn Reson Med. Vol. 24. 2016.

7. Maallo, Anne Margarette S., et al. "Effects of unilateral cortical resection of the visual cortex on bilateral human white matter." NeuroImage 207 (2020): 116345.

Citation
----
The paper is also available on arXiv: 
