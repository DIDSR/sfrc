import numpy as np 
import pydicom
import matplotlib.pyplot as plt
import imageio
import scipy.misc

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
    # input (min_data, max_data) with range (max_data - min_data) is normalized to (a, b)
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

def pydicom_imread(path):
  """ 
  reads image stored in a dicom file 
  in the dtype that was originally used to
  store the corresponding image array
  
  input
  -----
  path (str): dicom image path 
  
  output
  ------
  array in type foat32
  """
  input_image = pydicom.dcmread(path)
  return(input_image.pixel_array.astype('float32'))

def imageio_imread(path):
  """
  imageio based reading function that reads specified file
  in its original format even if its in - ve floats
  Note that the image data is returned as-is, and may not always 
  have a dtype of uint8
  """
  return(imageio.imread(path))

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
    customized version using matplotlib to imshow array as 2D image
    
    input
    -----
    arr   : 2D array
    xlabel: string to set x-coordinate label in the plot
    ylabel: string to set x-coordinate label in the plot
    title : string to set title in the plot
    cmap  : string to set colormap of the 2D plot
            'brg' is the optimal colormap for reb-green-blue image
            'brg_r': in 'brg' colormap green color area will have
             high values whereas in 'brg_r' blue area will have
             the highest values
             'Greys_r': is the default colormap
    colorbar: bool if true diplays colorbar
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
    customized function to show different layers of a 3D array 
    as 2D subplots
    
    usage
    ------
    multi2dplots(1, 2, lena_stack, axis=0, passed_fig_att={'colorbar': False, \
    'split_title': np.asanyarray(['a','b']),'out_path': 'last_lr.tif'})
    where lena_stack is of size (2, 512, 512)
    
    input
    -----
    nrows       : no. of rows in the subplots
    ncols       : no. of columns in the subplots
    fig_arr     : 3D array used for the subplots
    axis        : axis that is held constant and the 2D plot is demostrated
                  along the other two axes
    passed_fig_att : customized arguments imported from pyplot's subplot kwargs
                     See the default arguments below
                     
    output
    -----
    subplots as figure
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