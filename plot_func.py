#-----------------------------------------------
# @author: pkc 
#
# plot_func.py 
# ............
# includes functions used in plotting
#
import numpy as np 
import pydicom
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from shapely import LineString, get_coordinates
import sys

def add_subplot_border(ax, width=1, color=None ):
    """
    function used to add border to a given 
    axes in a subplot
    
    input
    -----
    ax  : usage ax[ith_row, jth_col] such that the indicated 
          [ith_row, jth_co] index with depict a bounded border 
          in the subplot
    width: width of the bounded border
    color: string on the color of the border
    
    output
    -----
    indicated axes interm of ax[i_th row, j_th col] will
    have a border
    """
    fig = ax.get_figure()

    # Convert bottom-left and top-right to display coordinates
    x0, y0 = ax.transAxes.transform((0, 0))
    x1, y1 = ax.transAxes.transform((1, 1))

    # Convert back to Axes coordinates
    x0, y0 = ax.transAxes.inverted().transform((x0, y0))
    x1, y1 = ax.transAxes.inverted().transform((x1, y1))

    rect = plt.Rectangle(
        (x0, y0), x1-x0, y1-y0,
        color=color,
        transform=ax.transAxes,
        zorder=-1,
        lw=2*width,
        fill=None,
    )
    fig.patches.append(rect)

def find_earliest_lp_intersection(x_coord, line1, line2):
    """
    intakes two line plots, plotted over the same x-coordinate.
    Then outputs the y-coordinate where the two lines intersect. 
    In case they intersect at multiple points, the y-coordinate 
    corresponding to the minimum x-coordinate is returned 
    
    input
    -------
    x_coord: 1d arrat
    line1: y values of the first line
    line2: y values of the second line 
    
    output
    -----
    single valued y-coordinate where the two line
    intersect
    """
    first_line   = LineString(np.column_stack((x_coord, line1)))
    second_line  = LineString(np.column_stack((x_coord, line2)))
    intersection = first_line.intersection(second_line)
    coordinates  = get_coordinates(intersection, return_index=False)

    if coordinates.size ==0:        
      # case where intersection does not happen
      xcoord_of_intersection = np.infty
    else:
      # return x-coordinate corresponding to the earlier intersection
      if intersection.geom_type == 'MultiPoint':
        xcoord_of_intersection = min(coordinates[:,0])
      elif intersection.geom_type == 'Point':
        xcoord_of_intersection = coordinates[0,0]
      else:
        print('ERROR! geom point of intersection beyond neither a point or a multipoint array')
        print('Look inside the function find_earliest_lp_intersection.')
        sys.exit()
    return (xcoord_of_intersection)

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

def dict_plot_of_2d_arr(args, rows, cols, bool_mat, arr_2d, cmap='Greys_r', save_plot=False, disp_plot=False, output_path='', plt_title=None):
  """
  customized version of pyplot's subplot function that displays and saves
  subplots with bounded red box on the subplots indicated using a boolean
  matrix
  
  input
  ------ 
  args      :parser.parse_ags()
             makes use of command line arguments on dtype and windowing
             used in CT imaging. For a more info on windowing review  
             https://radiopaedia.org/articles/windowing-ct?lang=us#:~:text=The%20window%20level%20(WL)%2C,be%20brighter%20and%20vice%20versa. 
  rows      :number of rows of subplots
  cols      :number of columns of subplots
  arr_2d    :stacked 2d arrays
  bool_mat  :a boolean matrix with 0s and 1s. 
             subplot whose corresponding value is 1 will be borded
             with a bounding box. 
  """
  # rows, cols indicate number of subplots along rows & columns
  # rows*cols = len(arr_2d)
  
  # because reading the LDGC CT data yields air as 0 HU,
  # the intercept/bias value below scales the air contrast
  # back to -1024 HU. 
  if args.in_dtype=='uint16':
    rescale_intercept = -1024
  else:
    rescale_intercept = 0.0

  if args.windowing=='soft':
    level = 50
    win   = 400
    win_max = (level - rescale_intercept) + win/2.0
    win_min = (level - rescale_intercept) - win/2.0
  elif args.windowing=='lung':
    level = -600
    win   = 1500
    win_max = (level - rescale_intercept) + win/2.0
    win_min = (level - rescale_intercept) - win/2.0
  elif args.windowing=='bone':
    level   = 400
    win     = 1800
    win_max = (level - rescale_intercept) + win/2.0
    win_min = (level - rescale_intercept) - win/2.0
  elif args.windowing=='unity':
    arr_2d = arr_2d/args.img_list_max_val
    win_min = 0.0
    win_max = 1.0
  elif args.windowing=='gray':
    arr_2d  = 255.0*arr_2d/args.img_list_max_val
    win_min = 0.0
    win_max = 255.0
  else:
    win_max=None
    win_min=None
  
  fig, ax = plt.subplots(nrows=rows,ncols=cols, figsize=(14, 14))# plt.figure(figsize=(14, 14)) # width height
  for i, comp in enumerate(arr_2d):
      plt.subplot(rows, cols, i + 1)
      plt.imshow(comp, cmap=cmap, interpolation="nearest", vmin=win_min, vmax=win_max)# vmin=win_min, vmax=win_max)# norm=colors.Normalize(vmin=0, vmax=1.0))#vmin=0, vmax=1.0)# vmin=825.0, vmax=1275.0)
      plt.xticks(())
      plt.yticks(())
  
  hallu_ind = np.where(bool_mat==1.0)

  if plt_title!=None: plt.suptitle(plt_title)
  #print(hallu_ind)
  if len(hallu_ind[0])!=0:
    for i in range(len(hallu_ind[0])):
        ax_i = hallu_ind[0][i]
        ax_j = hallu_ind[1][i]
        add_subplot_border(ax[ax_i, ax_j], 3, 'red')
  
  #plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.1)
  #plt.clim(0,1)
  if save_plot: plt.savefig(output_path)
  if disp_plot: plt.show()

def plot_n_save_img(threshold, fx_coord, frc_val, tx_coord, thres_val, output_img_name=None, save_img=True, display_img=False, plt_title='', mtf_space=False):  
    """
    plots, displays and save FRC curve of a full-sized image pairs
    
    input
    ------
    threshold      : string on the threshold used. Eg 'half-bit'
                     '0.5', '0.75'. It is used to set 
                     label for the threshold used in the legend of 
                     the full-image-based FRC plot. 
                     If string 'all' is used, then the legend is set
                     as 'one-bit', 'half-bit', 'EM' and 'num-constant'.
                    'all' is used as a helping option to see all the
                     threshold cutoff at the time of the code development.
    fx_coord       : 1d array of x-coordinate used in the frc plot
    frc_val        : 1d array of FRC value used in the FRC plot
    tx_coord       : 1d array of x-coordinate corresponding to threshold values
                     used in the frc plot. it is same as fx_coord
    thres_val      : 1d array or a list with 4 arrays corresponding to the 
                     four threshold: 'one-bit, half-bit, EM, 0.5. 
    save_img       : (bool) with True or False. If true then the FRC plot is
                     saved.
    output_img_name: (str) and If!=None and save_img option is True, 
                     then save the FRC plot with 
                     the filename supplied through this option.
    display_img    : displays the FRC plot as a GUI output
    plt_tile       : string as the title of the plot
    mtf_space      : bool (True or False). If true then the rings in the FRC have 
                     0.5 as center. Else, the rings will have 0.0 as its center. mtf_space
                     accounts for the pixel spacing. Look in the main.py file description. 
                     and in the function patchwise_frc where info on pixel spacing is 
                     incorporated when determining x-coordinates
    
    output
    ------
    """
    # if FRC space is true
    if mtf_space!=True: 
        frc_space=True
    else:
        frc_space=False

    if frc_space:
        # then x-coording corresponds to frc-space
        # with the range (0, 1) compress to (0, 0.5)
        fx_coord = fx_coord/2.0
        tx_coord = tx_coord/2.0
    # else no compression is required for the FRC space

    if threshold=='all':
        plt.plot(fx_coord[:-1], frc_val[:-1], label = 'chip-FRC', color='black')
        plt.plot(tx_coord[:-1], (thres_val[0])[:-1], label='one-bit', color='green')
        plt.plot(tx_coord[:-1], (thres_val[1])[:-1], label='half-bit', color='red')
        plt.plot(tx_coord[:-1], (thres_val[2])[:-1], label='EM', color='Orange')
        plt.plot(tx_coord[:-1], (thres_val[3])[:-1], label='num-constant', color='brown')

    else:
        plt.plot(fx_coord[:-1], frc_val[:-1], label = 'FRC', color='black')
        plt.plot(tx_coord[:-1], thres_val[:-1], label=threshold, color='red')

    plt.ylim(0.0, 1)
    if frc_space:
        plt.xlim(0, 0.5)
        plt.grid(linestyle='dotted', color='black', alpha=0.3) 
        plt.xticks(np.arange(0.0, 0.5, step=0.03))

    if mtf_space:
        plt.xlim(0, 1.0)
        plt.grid(linestyle='dotted', color='black', alpha=0.3) 
        plt.xticks(np.arange(0.0, 1.0, step=0.06))

    plt.yticks(np.arange(0, 1, step=0.1))
    plt.legend(prop={'size':13})
    plt.xlabel('Spatial Frequency (unit$^{-1}$)', {'size':13})
    plt.title (plt_title, {'size':10})
    plt.tick_params(axis='both', labelsize=7)
    if display_img: plt.show()
    if save_img: plt.savefig(output_img_name); plt.close()

def dict_plot_of_patched_frc(bool_hallu_1darr, args, fx_coord, stacked_frc, tx_coord, thres_val, output_img_name=None, save_img=False, display_img=False, plt_title=None, mtf_space=False):
    """
    customized version of pyplot's subplot function that displays, saves
    subplots patch-wise sFRC plot for a given image pair. It also outputs
    a boolean 2D matrix with 1's corresponding to the 
    patches that are hallucinated as determined using sFRC curve and the 
    hallucination threshold ht. Its other
    output is the sum of the boolean matrix, i.e., no of fakes for a given 
    image-pair
    
    input 
    ----
    bool_hallu_1darr: a boolean 1d array with 0s and 1s.
                      In sFRC calculations, this 1d input vector
                      is initialized with 1's along the patches 
                      where there is enough contrast to be hallucinated 
                      i.e., patches that pass air_threshold. Look in the function 
                      patchwise_sfrc in the file utils.py
    args            : parser.parse_ags()
                      this function makes use of command line arguments on 
                      threshold and ht. 
    fx_coord        : 1d array of x-coordinate used in the sfrc plot
    frc_val         : 1d array of FRC value used in the FRC plot
    tx_coord        : 1d array of x-coordinate corresponding to threshold values
                      used in the sfrc plot. it is same as fx_coord
    thres_val       : 1d array or a list with 4 arrays corresponding to the 
                      four threshold: 'one-bit, half-bit, EM, 0.5. 
    save_img        : (bool) with True or False. If true then the FRC plot is
                      saved.
    output_img_name : (str) and If!=None and save_img option is True, 
                      then save the FRC plot with 
                      the filename supplied through this option.
    display_img     : displays the FRC plot as a GUI output
    plt_title       : string as the title of all the subplots (on the top).
    mtf_space       : bool (True or False). If true then the rings in the FRC have 
                      0.5 as center. Else, the rings will have 0.0 as its center. mtf_space
                      accounts for the pixel spacing. Look in the main.py file description. 
                      and in the function patchwise_frc where info on pixel spacing is 
                      incorporated when determining x-coordinates
    use direction
    -------------
    (1) enumerate lists array row-wise. So different patch-based frc should be stacked row-wise  
    (2) row and cols of subplots is dictated by no of patches used to formulated stacked_frc arr.
        i.e., height of the arr
    
    outputs
    ------
    (1) patch-wise sFRC plots either display and/or saved as image
    (2) boolean 2d matrix and the sum of the boolean matrix
    """
    # rows, cols indicate number of subplots along rows & columns
    # rows*cols = len(arr_2d)
    Npatches, _    = stacked_frc.shape
    Nrows          = int(np.ceil(np.sqrt(Npatches)))
    Ncols          = Nrows
    bool_hallu_mat = np.reshape(bool_hallu_1darr, (Nrows, Ncols))
    plt.figure(figsize=(16, 12)) # width, height

    # if FRC space is true
    if mtf_space!=True: 
        frc_space=True
    else:
        frc_space=False

    if frc_space:
        # then x-coordinate corresponds to frc-space
        # with the range (0, 1) compress to (0, 0.5)
        fx_coord = fx_coord/2.0
        tx_coord = tx_coord/2.0
    # else no compression is required for the FRC space

    # plotting attributed to each patch-based subplots
    patch_ind = 0
    for ii in range(Nrows):
        for jj in range (Ncols):
            frc_val = stacked_frc[patch_ind,:]
            plt.subplot(Nrows, Ncols, patch_ind + 1)
            if args.frc_threshold=='all':
                plt.plot(fx_coord[:-1], frc_val[:-1], label = 'FRC', color='black')
                plt.plot(tx_coord[:-1], (thres_val[0])[:-1], label='one-bit', color='green')
                plt.plot(tx_coord[:-1], (thres_val[1])[:-1], label='half-bit', color='red')
                plt.plot(tx_coord[:-1], (thres_val[2])[:-1], label='0.5 -Thres', color='brown')
                plt.plot(tx_coord[:-1], (thres_val[3])[:-1], label='EM', color='Orange')
            else:
                plt.plot(fx_coord[:-1], frc_val[:-1],   label = 'FRC', color='black')
                plt.plot(tx_coord[:-1], thres_val[:-1], label=args.frc_threshold, color='red')
            
            plt.grid(linestyle='dotted', color='black', alpha=0.3) 
            
            # y tick shown only on the first column-based subplots
            if (jj!=0): 
                plt.yticks(np.arange(0, 1, step=0.25), visible=False)
            else:
                plt.yticks(np.arange(0, 1, step=0.25))

            # x tick show only on last row
            if(ii!=(Nrows-1)):
                plt.xticks(np.arange(0.0, 0.5, step=0.05), visible=False)
            else:
                plt.xticks(np.arange(0.0, 0.5, step=0.05), rotation=90)

            plt.ylim(0.0, 1)
            if frc_space:
                plt.xlim(0, 0.5)
                plt.grid(linestyle='dotted', color='black', alpha=0.3) 
                plt.xticks(np.arange(0.0, 0.5, step=0.05))

            if mtf_space:
                plt.xlim(0, 1.0)
                plt.grid(linestyle='dotted', color='black', alpha=0.3) 
                plt.xticks(np.arange(0.0, 1.0, step=0.1))

            plt.yticks(np.arange(0, 1, step=0.25))
            #plt.legend(prop={'size':13})
            #plt.xlabel('Spatial Frequency (unit$^{-1}$)', {'size':13})
            #plt.title (plt_title, {'size':10})
            #plt.tick_params(axis='both', labelsize=7)

            if args.frc_threshold!='all':
                xcoord_of_frc_n_th_intersection = find_earliest_lp_intersection(fx_coord[:-1], frc_val[:-1], thres_val[:-1])

                # patches where there is no intersection or intersection is above the hallucination threshold
                # are re-designated as False boolean value
                if (bool_hallu_mat[ii, jj]==1.0) and (xcoord_of_frc_n_th_intersection>args.ht):
                    bool_hallu_mat[ii, jj]=0.0
                #if (bool_hallu_mat[ii, jj]==1.0) and (xcoord_of_frc_n_th_intersection<0.45 or xcoord_of_frc_n_th_intersection>0.51):
                #    bool_hallu_mat[ii, jj]=0.0
            patch_ind = patch_ind+1

    # No of patches labeled as fake on an image        
    per_img_pair_fk = np.sum(bool_hallu_mat)
             
    # print(bool_hallu_mat)
    # general attributes on the whole plot
    plt.subplots_adjust(wspace=0.08, hspace=0.1)
    if plt_title!=None: plt.suptitle(plt_title)
    if display_img: plt.show()
    if save_img: plt.savefig(output_img_name); plt.close()

    return(bool_hallu_mat, per_img_pair_fk)
