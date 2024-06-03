.. raw:: html

    <p align="center"><img src="paper_plots/intro_plot.png" alt="Logo" width="600"/></p>

sFRC for detecting fakes in AI-assisted medical image restoration (postprocessing or reconstruction) 
======================================================================================================

- **sFRC**: scans and performs Fourier Ring Correlation (FRC)-based analysis over small patches between images from AI-assisted methods and their reference counterparts to objectively and automatically identify fakes as detailed in our 
  `sFRC paper <https://www.techrxiv.org/users/763069/articles/740286-fake-detection-in-ai-assisted-image-recovery-using-scanning-fourier-ring-correlation-sfrc>`_. You can also perform sFRC analysis to find fakes from traditional regularization-based methods by simply comparing images from regularization-based vs. reference methods. 
- **Inputs**: Restored medical images from Deep learning- or Regularization-based methods and their reference counterparts from the standard-of-care methods (such as FBP), and hallucination threshold.
- **Outputs**: Small-sized red bounding boxes on input images that are deemed as fake ROIs (in AI-assisted as well as reference images), and the total number of such fake ROIs in the provided input images. 
  Demo movie files on sFRC-labeled fakes for a CT super-resolution problem is provided `here <https://fdahhs.ent.box.com/s/vvfcbqxd66a2x09yld1tyk2weqs72i7s>`_.
- **Demo**: On two image restoration problems: CT super-resolution (**ctsr**), and MRI sub-sampling (**mrsub**).

.. contents::

Usage
-----

.. code-block::

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
    --patch-size          p96 or p64 or 48 or p32 to indicate patch sizes for the sFRC analysis. Change padding option in main.py for a
                          different patch size.
    --random_N            performs sfrc calculation on randomly selected 16 complimentary images from DL/Reg - Reference folders.
                          For more info refer to in-built options in main.py.
    --input-gen-folder    folder name containing DL or regularization method-based outputs.
    --target-gen-folder   folder name containing reference method-based outputs.
    --img-format          image format for input and target images. Dicom/raw/tif/png? To add a new image format read function look 
                          inside the function partition_read_normalize_n_augment in file mpi_utils.py.
    --multi-patients      if there are multiple-subfolders related to different parents.
    --remove-edge-padding remove patches at the edges of images when mod(img size, patch size) != 0.
    --apply-hann          apply hanning filter before the frc calculation.
    --frc-threshold       frc threshold to determine correlation cut-off between the 2 methods. This patch-based FRC analysis
                          is better suited with a constant threshold such as 0.5, 0.75. Other common options include half-bit, all,
                          one-bit. To add new threshold, look inside function FRC in the file frc_utils.py.
    --inscribed-rings     max frequency at which correlation is calculated is img (or patch) length/2. if false then frc will be
                          calculated upto the corner of the image (or patch).
    --anaRing             perimeter of circle-based calculation to determine data points in each ring. Otherwise, no. of pixels in
                          each ring used to determine data points in each ring.
    --rNx RNX             image x-size for raw image as input.
    --rNy RNY             image y-size for raw image as input. Default is same dim as rNx
    --in-dtype            data type of input images. It is needed for images with .raw filenames. It is also needed to set the maximum 
                          intensity value for air thresholding and windowing of patches when saving bounding box-based outputs.
    --save-patched-subplots
                          if you want to save patches with the bounding box and FRC plot results.
    --apply-bm3d          apply image-based mild bm3d smoothing before the frc calculation. It decreases the chance of quick FRC
                          drop. which means it increases the chance of missing fake artifacts. But it has advantage of increasing PPV.
    --mtf-space           x-axis for FRC is in the mtf space. Uses the dx info. Use this option only if you have info on dx for your
                          acquisition. Otherwise, do not use this option. When this option is not used, x-axis for FRC has unit pixel(^-1).
    --dx                  xy plane pixel spacing. Default value is set from the LDGC dataset and has the unit mm.
    --ht                  patches whose x-coordinates (corresponding to the points when their FRC curves intersect with the frc-
                          threshold) that fall below this ht threshold will be labeled as fake ROIs.
    --windowing           windowing used when generating the patched subplots Options include soft, lung, bone, unity and none.
                          Setting appropriate viewing window is very important in zeroing anomalies between a DL method- and
                          reference method-based outputs. For a sanity check, you may choose to confirm the marked ROIs generated
                          from this implementation by using software like ImageJ under different type of windowing.
    --remove-ref-noise    applies a gentle bilateral filtering to reference images.
    --img-y-padding       pads y-dim with zeros with pad_width=(rNx-rNy). It is useful when analyzing coronal-slices.

|

Requirements
------------
Install `openmpi <https://www.open-mpi.org/>`_ if your machine does not have one. A guide is provided in the file
./requirements/openmpi_setup.txt. Export paths related to openmpi's compilers and libraries 
as your environment variable as follows:

.. code-block::
     
     $ export PATH=$HOME/path/to/openmpi/bin:$PATH
     $ export LD_LIBRARY_PATH=$HOME/path/to/openmpi/lib:$LD_LIBRARY_PATH
     
Create a new conda enviroment and install the required packages as follows:

.. code-block::
    
    $ conda create -n mpi_sfrc python=3.7.5 --no-default-packages
    $ conda activate mpi_sfrc
    $ conda install -c anaconda h5py==3.6.0
    $ pip install -r ./requirements/sfrc_requirements.txt
    $ pip install -r ./requirements/mri_unet_requirements.txt

|

DEMO execution of sFRC
----------------------------------------------------------
The example codes below show how to run sfrc by using data from DL/Reg methods and their reference counterparts used in our paper. 
Run the codes below. Then accordingly change input paths and sfrc parameters for your application. 

1. sFRC on SRGAN-based CT upsampled (x4) outputs

   .. code-block::
      
      OUTPUT_FNAME="./results/CT/sm_srgan_sel_sh_L067/"
      INPUT_FOLDER="./ctsr/results/test_sh_L067/ua_ll_smSRGANsel_in_x4/checkpoint-generator-20/"
      INPUT_GEN="test_sh_L067_cnn"
      TARGET_GEN="test_sh_L067_gt"
      time mpirun --mca btl ^openib -np 1 python main.py --input-folder ${INPUT_FOLDER} --output-folder ${OUTPUT_FNAME} --patch-size 'p64'  --input-gen-folder ${INPUT_GEN} --target-gen-folder ${TARGET_GEN} --img-format 'raw' --frc-threshold '0.5' --in-dtype 'uint16' --anaRing --inscribed-rings --rNx 512 --apply-hann --mtf-space --ht 0.33 --windowing 'soft' --save-patched-subplots
   
   OR execute the demo bash file
   
   .. code-block:: 
      
      bash +x demo_sfrc_run.sh 'CT' 'sel' 'sh' 1

   'CT' indicates sfrc on CT-based data. 'sh' and 'sel' are options to indicate paths for sharp kernel-based data and 
   tuning set for sFRC parameters used in our paper. Likewise 'sm' indicates smooth kernel-based test set. 
   1 indicates one processing unit (-np) to be used in our mpi-based sFRC implementation. 
   Note that, in this git repo, the demo example for the CT application includes only 5 CT images. 
   As such, the no. of fakes, for the specified parameters, for sharp and smooth data will be 21 
   and 16 respectively. Refer to the next subsection to fetch the complete test set and results as 
   provided in our paper for the CT application. 

2. sFRC on UNet- and PLSTV-based MRI outputs from a subsampled acquisition (x3)

   .. code-block::
      
      bash +x demo_sfrc_run.sh 'MRI' '' 'unet' 4

   Change the third option to 'plstv' for the plstv-based results provided in our paper. 

|

Apply trained SRGAN 
--------------------
The SRGAN checkpoint provided in this repository was trained using CT images from the six patients provided in 
`LDGC dataset <https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026>`_ and as detailed in our paper.
This checkpoint can be applied to the low-resolution CT images provided in this repository in the following manner: 

3. sFRC on SRGAN (tuning set)

   .. code-block:: 

      cd ctsr
      bash +x demo_srgan_test.sh 'sel' 'sh' #on sharp kernel-based tuning set

To apply the SRGAN to all the CT images from patient L067 (as described in our paper) refer to "./ctsr/create_sr_dataset/readme.txt".
Once you successfully download and preprocess smooth and sharp CT scans corresponding to patient L067, the following commands will 
yield fake patches as tabulated in TABLE I in our paper and as depicted in the following 
`movie files <https://fdahhs.ent.box.com/s/vvfcbqxd66a2x09yld1tyk2weqs72i7s>`_.

4. sFRC on SRGAN (test set)

   .. code-block:: 

      cd ctsr
      bash +x demo_srgan_test.sh '' 'sh'
      bash +x demo_srgan_test.sh '' 'sm'

Then set the first command line input as a blank string, '', to indicate tags related to the paths 
of CT images are test set for the sFRC analysis (as used in our paper) when executing demo_sfrc_run.sh.

.. code-block:: 

   cd ..
   bash +x demo_sfrc_run.sh '' 'sh' 47 # on sharp test data with 47 set as no. of processors
   bash +x demo_sfrc_run.sh '' 'sm' 47 #on smooth test data with 47 set as the no. of processors

|

Apply trained UNet 
-------------------
The trained Unet model and data provided in this repository (as well as used in our paper) have been imported from the following github
repository: `hallucinations-tomo-recon <https://github.com/comp-imaging-sci/hallucinations-tomo-recon>`_. Also, 
`Pediatric epilepsy resection MRI dataset <https://kilthub.cmu.edu/articles/dataset/Pediatric_epilepsy_resection_MRI_dataset/9856205>`_ is 
the original source of the MRI data. 

.. code-block:: 
   
   cd mrsub/unet
   bash +x run_unet_test.sh
|

PLSTV-based reconstruction 
-------------------------------
Follow the installation instructions provided in the `BART repository <https://mrirecon.github.io/bart/>`_.
Then edit the path to BART's python wrapper in line 20 in file "./mrsub/plstv/bart_pls_tv.py".

.. code-block:: 

   cd mrsub/plstv
   bash +x run_bart_pls_tv.sh

|

References 
----------
1. McCollough, Cynthia H., et al. "Low‐dose CT for the detection and classification of metastatic liver lesions: results of the 2016 low dose CT grand challenge." Medical physics 44.10 (2017): e339-e352.

2. Bhadra, Sayantan, et al. "On hallucinations in tomographic image reconstruction." IEEE transactions on medical imaging 40.11 (2021): 3249-3260.

3. `hallucinations-tomo-recon <https://github.com/comp-imaging-sci/hallucinations-tomo-recon>`_.

4. Ledig, Christian, et al. "Photo-realistic single image super-resolution using a generative adversarial network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

5. Sergeev, Alexander, and Mike Del Balso. "Horovod: fast and easy distributed deep learning in TensorFlow." arXiv preprint arXiv:1802.05799 (2018).

6. Uecker, Martin, et al. "The BART toolbox for computational magnetic resonance imaging." Proc Intl Soc Magn Reson Med. Vol. 24. 2016.

7. Maallo, Anne Margarette S., et al. "Effects of unilateral cortical resection of the visual cortex on bilateral human white matter." NeuroImage 207 (2020): 116345.

8. Maallo, Anne; Liu, Tina; Freud, Erez; Patterson, Christina; Behrmann, Marlene (2019). Pediatric epilepsy resection MRI dataset. Carnegie Mellon University. Dataset. https://doi.org/10.1184/R1/9856205.

|

License and Copyright
---------------------------
This software and documentation (the "Software") were developed at the Food and Drug Administration (FDA) by employees of the Federal Government in the course of their official duties. Pursuant to Title 17, Section 105 of the United States Code, this work is not subject to copyright protection and is in the public domain. 
Permission is hereby granted, free of charge, to any person obtaining a copy of the Software, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software or derivatives, and to permit persons to whom the Software is furnished to do so. FDA assumes no responsibility whatsoever for use by other parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. Further, use of this code in no way implies endorsement by the FDA or confers any advantage in regulatory decisions. Although this software can be redistributed and/or modified freely, we ask that any derivative works bear some notice that they are derived from it, and any modified versions bear some notice that they have been modified.

|

Citation
--------

::

   @article{kc2024fake,
     title={Fake detection in AI-assisted image recovery using scanning Fourier Ring Correlation (sFRC)},
     author={Kc, Prabhat and Zeng, Rongping and Soni, Nirmal and Badano, Aldo},
     journal={TechRxiv Preprints},
     year={2024},
     doi={10.36227/techrxiv.171259560.02243347/v1},
   }

|

Contact
--------
prabhat.kc@fda.hhs.gov
