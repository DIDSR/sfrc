sFRC for detecting fakes in AI-assisted medical image recovery
=========================================================================
This implementation performs Fourier Ring Correlation (FRC)-based analysis over small patches and concomitantly (s)canning
across Deep learning(DL) or regularization(Reg)-based outputs and their reference counterparts to identify fakes.

1. DL/Reg method- and Reference method-based data for sFRC 
==============================

1. Get SRGAN-based CT upsampled (x4) output
==============================

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
=========================================================================
All the post-processing codes, data have been sourced from . Other packages such as BART and fastmri are 

Usage::

  python main_3d.py --acceleration_factor 4

edit the path to BART's python wrapper in line 20 in file mrsub/plstv/bart_pls_tv.py
  cd mrsub/unet
  chmod +x run_unet_test.sh
  ./run_unet_tesh.sh
----

3. sFRC analysis on the SRGAN-based outputs
=========================================================================

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
=========================================================================

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
----

The paper is also available on arXiv: 
