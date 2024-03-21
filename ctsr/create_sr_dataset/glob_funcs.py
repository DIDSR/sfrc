import numpy as np 
import pydicom
import matplotlib.pyplot as plt
import imageio
import scipy.misc

def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
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


def normalize_data_ab(a, b, data):
    # input (min_data, max_data) with range (max_data - min_data) is normalized to (a, b)
    min_x = min(data.ravel())
    max_x = max(data.ravel())  
    range_x = max_x - min_x 
    return((b-a)*((data-min_x)/range_x)+a)

def normalize_data_ab_cd(a, b, c, d, data):
    # input data (min_data, max_data) with range (d-c) is normalized to (a, b)
    min_x = c
    max_x = d  
    range_x = max_x - min_x 
    return((b-a)*((data-min_x)/range_x)+a)

def add_rnl_white(rnl, b, mu, sigma):
    """ sigma = std 
    var = sigma^2
    """
    h, w = b.shape
    randn =  np.random.normal(loc = mu, scale = sigma, size = (h,w))
    e = randn/np.linalg.norm(randn, ord = 2)
    e = rnl*np.linalg.norm(b)*e;
    return(b + e)

def pydicom_imread(path):
  """ reads dicom image with filename path 
  and dtype be its original form
  """
  input_image = pydicom.dcmread(path)
  return(input_image.pixel_array.astype('float32'))

def imageio_imread(path):
  """
   imageio based imread reads image in its orginal form even if its in
   - ve floats
  """
  return(imageio.imread(path).astype('float32'))

def imsave(image, path, type=None):
  
  """
    imageio will save values in its orginal form even if its float
    if type='original' is specified
    else scipy save will save the image in (0 - 255 values)
    scipy new update has removed imsave from scipy.misc due
    to reported errors ... so just use imwrite from imageio 
    by declaring orginal and changing the data types accordingly
  """
  if type is "original":
    return(imageio.imwrite(path, image))
  else:
    return scipy.misc.imsave(path, image)

def plot2dlayers(arr, xlabel=None, ylabel=None, title=None, cmap=None, colorbar=True):
    """
    'brg' is the best colormap for reb-green-blue image
    'brg_r': in 'brg' colormap green color area will have
        high values whereas in 'brg_r' blue area will have
        the highest values
    """
    if xlabel is None:
        xlabel=''
    if ylabel is None:
        ylabel=''
    if title is None:
        title=''
    if cmap is None:
        cmap='Greys_r'
    plt.imshow(arr, cmap=cmap)
    cb = plt.colorbar()
    if colorbar is False:
      cb.remove()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    

def multi2dplots(nrows, ncols, fig_arr, axis, passed_fig_att=None):
    """
      gf.multi2dplots(1, 2, lena_stack, axis=0, passed_fig_att={'colorbar': False, 'split_title': np.asanyarray(['a','b']),'out_path': 'last_lr.tif'})
      where lena_stack is of size (2, 512, 512)
    """
    default_att= {"suptitle": '',
            "split_title": np.asanyarray(['']*(nrows*ncols)),
            "supfontsize": 12,
            "xaxis_vis"  : False,
            "yaxis_vis"  : False,
            "out_path"   : '',
            "figsize"    : [8, 8],
            "cmap"       : 'Greys_r',
            "plt_tight"  : True,
            "colorbar"   : True
                 }
    if passed_fig_att is None:
        fig_att = default_att
    else:
        fig_att = default_att
        for key, val in passed_fig_att.items():
            fig_att[key]=val
    
    f, axarr = plt.subplots(nrows, ncols, figsize = fig_att["figsize"])
    img_ind  = 0
    f.suptitle(fig_att["suptitle"], fontsize = fig_att["supfontsize"])
    for i in range(nrows):
        for j in range(ncols):                
            if (axis==0):
                each_img = fig_arr[img_ind, :, :]
            if (axis==1):
                each_img = fig_arr[:, img_ind, :]
            if (axis==2):
                each_img = fig_arr[:, :, img_ind]
                
            if(nrows==1):
                ax = axarr[j]
            elif(ncols ==1):
                ax =axarr[i]
            else:
                ax = axarr[i,j]
            im = ax.imshow(each_img, cmap = fig_att["cmap"])
            if fig_att["colorbar"] is True:  f.colorbar(im, ax=ax)
            ax.set_title(fig_att["split_title"][img_ind])
            ax.get_xaxis().set_visible(fig_att["xaxis_vis"])
            ax.get_yaxis().set_visible(fig_att["yaxis_vis"])
            img_ind = img_ind + 1
            if fig_att["plt_tight"] is True: plt.tight_layout()
            
    if (len(fig_att["out_path"])==0):
        plt.show()
    else:
        plt.savefig(fig_att["out_path"])