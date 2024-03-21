import numpy as np
import numpy.fft as fft
import frc_utils
import io_func
import frc_utils
import plot_func as pf 
import os
import utils

def get3_comp_of_img(img, band1_ep, band2_ep, band3_ep):
    h, w   = img.shape
    img_p1 = np.zeros([h,w]).astype('complex')
    img_p2 = np.zeros([h,w]).astype('complex')
    img_p3 = np.zeros([h,w]).astype('complex')
    img_fft = fft.fftshift(fft.fft2(img))

    indices=frc_utils.ring_indices(img_fft)
    for i in range(0, band1_ep):
        img_p1[indices[i]]=img_fft[indices[i]]
    for i in range(band1_ep, band2_ep):
        img_p2[indices[i]]=img_fft[indices[i]]
    for i in range(band2_ep, band3_ep):
        img_p3[indices[i]]=img_fft[indices[i]]

    img_p1 = np.real(fft.ifft2(fft.ifftshift(img_p1)))
    img_p2 = np.real(fft.ifft2(fft.ifftshift(img_p2)))
    img_p3 = np.real(fft.ifft2(fft.ifftshift(img_p3)))
    img_stack = np.stack((img.reshape(h,w), img_p1.reshape(h, w), img_p2.reshape(h, w), img_p3.reshape(h, w)), axis=0)
    return(img_p1, img_p2, img_p3, img_stack)

def get3_fft_bands_of_img(img, band1_ep, band2_ep, band3_ep):
    h, w   = img.shape
    img_p1 = np.zeros([h,w])
    img_p2 = np.zeros([h,w])
    img_p3 = np.zeros([h,w])
    img_fft = np.log(np.abs(fft.fftshift(fft.fft2(img))))

    indices=frc_utils.ring_indices(img_fft)
    for i in range(0, band1_ep):
        img_p1[indices[i]]=img_fft[indices[i]]
    for i in range(band1_ep, band2_ep):
        img_p2[indices[i]]=img_fft[indices[i]]
    for i in range(band2_ep, band3_ep):
        img_p3[indices[i]]=img_fft[indices[i]]
    img_stack = np.stack((img_fft.reshape(h,w), img_p1.reshape(h, w), img_p2.reshape(h, w), img_p3.reshape(h, w)), axis=0)
    return(img_p1, img_p2, img_p3, img_stack)

#plot and save logins
plot_fig = False
save_fig = True 
# crop img path
gt_path  = './paper_plots/plot2/crop_img_uint8_L_50_w_400/gt_000069.png'
cnn_path = './paper_plots/plot2/crop_img_uint8_L_50_w_400/srgan_000069.png'
out_path = './paper_plots/plot2/main_fig/'

gt_img  = io_func.imageio_imread(gt_path)
cnn_img = io_func.imageio_imread(cnn_path)

h, w   = gt_img.shape
r      = int(h/2)
fs_c1  = int(0.1*r) 
fs_c2  = int(0.25*r)

cnn_c1, cnn_c2, cnn_c3, cnn_c_stack = get3_comp_of_img(cnn_img, fs_c1, fs_c2, r)
gt_c1,  gt_c2,  gt_c3,  gt_c_stack  = get3_comp_of_img(gt_img, fs_c1, fs_c2, r)

cnn_fft_c1, cnn_fft_c2, cnn_fft_c3, cnn_fft_c_stack = get3_fft_bands_of_img(cnn_img, fs_c1, fs_c2, r)

if plot_fig:
    pf.multi2dplots(1, 4, cnn_c_stack, axis=0, passed_fig_att={'colorbar': False})
    pf.multi2dplots(1, 4, cnn_fft_c_stack, axis=0, passed_fig_att={'colorbar': False})
    pf.plot2dlayers(gt_img)

if save_fig:
    if not os.path.isdir(out_path): os.makedirs(out_path)
    # cnn part
    cnn_c1 = utils.normalize_data_ab(0, 255, cnn_c1).astype('uint8')
    cnn_c2 = utils.normalize_data_ab(0, 255, cnn_c2).astype('uint8')
    cnn_c3 = utils.normalize_data_ab(0, 255, cnn_c3).astype('uint8')
    
    #gt parts
    gt_c1 = utils.normalize_data_ab(0, 255, gt_c1).astype('uint8')
    gt_c2 = utils.normalize_data_ab(0, 255, gt_c2).astype('uint8')
    gt_c3 = utils.normalize_data_ab(0, 255, gt_c3).astype('uint8')
    #cnn fft parts
    cnn_fft_c1 = utils.normalize_data_ab(0, 255, cnn_fft_c1).astype('uint8')
    cnn_fft_c2 = utils.normalize_data_ab(0, 255, cnn_fft_c2).astype('uint8')
    cnn_fft_c3 = utils.normalize_data_ab(0, 255, cnn_fft_c3).astype('uint8')
    cnn_fft    = utils.normalize_data_ab(0, 255, cnn_fft_c_stack[0]).astype('uint8')
    io_func.imsave(cnn_c1, path=out_path + 'cnn_c1.png', svtype='original')
    io_func.imsave(cnn_c2, path=out_path + 'cnn_c2.png', svtype='original')
    io_func.imsave(cnn_c3, path=out_path + 'cnn_c3.png', svtype='original')

    io_func.imsave(gt_c1, path=out_path + 'gt_c1.png', svtype='original')
    io_func.imsave(gt_c2, path=out_path + 'gt_c2.png', svtype='original')
    io_func.imsave(gt_c3, path=out_path + 'gt_c3.png', svtype='original')
    
    io_func.imsave(cnn_fft_c1, path=out_path + 'cnn_fft_c1.png', svtype='original')
    io_func.imsave(cnn_fft_c2, path=out_path + 'cnn_fft_c2.png', svtype='original')
    io_func.imsave(cnn_fft_c3, path=out_path + 'cnn_fft_c3.png', svtype='original')
    io_func.imsave(cnn_fft, path=out_path    + 'cnn_fft.png', svtype='original')

