import sys
import os

sys.path.append('..')
import cv2
from src import io_func as io
from src import plot_func as pf

gt_path        = 'data/L506_sh_fd_slice_32/mk_L506_tv_slice_32.raw'
lr_out_dir     = 'results/blurr/input/'
scale          = 4
in_dtype       = 'uint16'
rNx            = 512

if not os.path.isdir(lr_out_dir): os.makedirs(lr_out_dir)
hr_image = io.raw_imread(gt_path, (rNx, rNx), in_dtype)
pf.plot2dlayers(hr_image, title='reference')
h, w  = hr_image.shape

lr_image = cv2.resize(hr_image, (int(w/scale), int(h/scale)),interpolation=cv2.INTER_AREA)
lr_image = lr_image.astype(in_dtype)
io.imsave_raw(lr_image, lr_out_dir + 'L506_tv_slice_32_down_4.raw' )
pf.plot2dlayers(lr_image, title='blurred img')