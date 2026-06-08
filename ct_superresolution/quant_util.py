import sys
#sys.path.append('/home/prabhat.kc/Implementations/python/')
#import global_files as gf
import numpy as np
import matplotlib.pyplot as plt
import os
import util
import io_func
from skimage.metrics import structural_similarity as compare_ssim
import torch

def relative_se(f_true, f_est):
    """
    https://lightning.ai/docs/torchmetrics/stable/regression/rse.html
    Computes the Relative Squared Error (RSE) between a true and 
    estimated array, normalized by the total squared deviation of the 
    true values from their mean.

    Parameters
    ----------
    f_true : numpy.ndarray
        The ground truth array of values.
    f_est : numpy.ndarray
        The estimated or predicted array of values, must be the same 
        shape as f_true.

    Returns
    -------
    float
        The relative squared error, representing the ratio of the 
        total squared error between f_true and f_est to the total 
        variance of f_true.
    """
    imdiff = f_true-f_est
    nume = np.sum(imdiff**2)
    deno = np.mean(f_true)-f_true
    deno = np.sum(deno**2)
    return(nume/deno)

def psnr(f_true, f_est, max_val=1.0):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    PSNR is a common metric used to measure the quality of a reconstructed
    or estimated image compared to a reference (ground truth) image.

    Parameters:
        f_true (ndarray): Ground truth (reference) image.
        f_est (ndarray) : Estimated or reconstructed image.
        max_val (float) : Maximum possible pixel value of the images
                         (default is 1.0 for normalized images).

    Returns:
        float: PSNR value in decibels (dB). Higher values indicate better
               similarity between the images.
    """    
    imdff = f_true - f_est
    rmse = np.sqrt(np.mean(imdff **2))
    psnr = 20.0*np.log10(max_val/rmse)
    return(psnr)

def quant_ana(output, target, img_type):
    """
    Performs quantitative analysis of a CNN model's output against a target 
    image by computing Peak Signal-to-Noise Ratio (PSNR) and Structural 
    Similarity Index (SSIM).

    The function handles preprocessing and type conversion of both the model 
    output and target tensors based on the specified image type, then returns 
    both metrics as float tensors.

    Parameters
    ----------
    output : torch.Tensor
        The CNN model's predicted output tensor of shape (B, C, H, W), where 
        B is batch size, C is number of channels, H is height, and W is width.
        The function processes only the first channel (index 0).
    target : torch.Tensor
        The ground truth image tensor of the same shape as output (B, C, H, W).
        The function processes only the first channel (index 0).
    img_type : str
        Specifies the image type, which determines how pixel values are scaled 
        and cast before metric computation. Accepted values:
            - 'natural'        : Converts output and target to uint8 in [0, 255]
                                 by scaling with a factor of 255.
            - 'natural-float'  : Normalizes output to float64 in [0, 255] using 
                                 util.normalize_data_ab; scales target to float64 
                                 in [0, 255].
            - 'positive-float' : Normalizes both output and target to float32 
                                 in [0, 1] using util.normalize_data_ab.
            - other (default)  : Treats values as general floats (possibly 
                                 negative); scales both output and target by 255 
                                 without normalization or clipping.

    Returns
    -------
    tuple of (torch.FloatTensor, torch.FloatTensor)
        A tuple containing:
            - _psnr : PSNR value (in dB) between the model output and target, 
                      returned as a float tensor. Higher values indicate better 
                      reconstruction quality.
            - _ssim : SSIM value between the model output and target, returned 
                      as a float tensor. Values range from -1 to 1, where 1 
                      indicates perfect structural similarity. SSIM is computed 
                      as an average across all channels (multichannel=True).

    Notes
    -----
    - Both output and target tensors are moved to CPU before processing.
    - The function uses util.normalize_data_ab for normalization in the 
      'natural-float' and 'positive-float' cases.
    - PSNR and SSIM are computed using external psnr() and compare_ssim() 
      functions, respectively.
    - when multichannel==true is set each channel will be processed independently 
      while determining their corresponding ssim values. The final output
      will be an average of all these channels 
    """    
    cnn_output = output.cpu()
    target     = target.cpu()
    
    cnn_output = cnn_output[:, 0, :, :].detach().numpy()
    cnn_output = cnn_output.transpose(1, 2, 0)

    target = target[:, 0, :, :].detach().numpy()
    target = target.transpose(1, 2, 0)
    #print(cnn_output.shape, target.shape)
    #print(cnn_output.shape, target.shape)
    #multi2dplots(4, 4, cnn_output, axis=0)
    #multi2dplots(4, 4, target, axis=0)

    # if true each channel will be processed independently while determining
    # the ssim values and finally out will be an average of all these 
    # channels 
    multichannel=True
    if img_type   == 'natural':
        model_out = np.uint8(cnn_output*255)
        target    = np.uint8(target*255)
        _psnr     = psnr((model_out), np.uint8(target), max_val=255)
        _ssim     = compare_ssim(model_out, np.uint8(target),  multichannel=multichannel, data_range=255)
    
    elif img_type == 'natural-float':
        model_out = np.float64(util.normalize_data_ab(0, 255, cnn_output))
        target    = np.float64(target*255)
        _psnr     = psnr((model_out), np.float64(target), max_val=255.0)
        _ssim     = compare_ssim(model_out, np.float64(target),  multichannel=multichannel, data_range=255)
    
    elif img_type == 'positive-float':
        
        model_out = np.float32(util.normalize_data_ab(0, 1, cnn_output))
        target    = np.float32(util.normalize_data_ab(0, 1, target))
        _psnr     = psnr(model_out, np.float32(target), max_val=1.0)
        _ssim     = compare_ssim(model_out, np.float32(target),  multichannel=multichannel, data_range=1)
    
    else: 	
        #just float (may have -ve values)
        model_out=(cnn_output*255)
        target   = target*255
        _psnr    = psnr(np.float64(model_out), np.float64(target), max_val=255.0)
        _ssim    = compare_ssim(model_out, np.float64(target),  multichannel=multichannel, data_range=255)

    _psnr = torch.tensor(_psnr)
    _ssim = torch.tensor(_ssim)
    return (_psnr.float(), _ssim.float())

