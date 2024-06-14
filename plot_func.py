#-----------------------------------------------
# @author: pkc 
#
# plot_func.py 
# ............
# includes functions used to read and write files
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
    """function used to add """
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
    Then outputs x-
        line1: y values of first line plot
    line2: y values of second line plot
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
    usagae : 
    multi2dplots(1, 2, lena_stack, axis=0, passed_fig_att={'colorbar': False, \
    'split_title': np.asanyarray(['a','b']),'out_path': 'last_lr.tif'})
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

def dict_plot_of_2d_arr(args, rows, cols, bool_mat, arr_2d, cmap='Greys_r', save_plot=False, disp_plot=False, output_path='', plt_title=None):
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
        plt.plot(tx_coord[:-1], (thres_val[2])[:-1], label='0.5 -Thres', color='brown')
        plt.plot(tx_coord[:-1], (thres_val[3])[:-1], label='EM', color='Orange')
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
    '''
    (1) enumerate lists array row-wise. So different patch-based frc should be stacked row-wise  
    (2) row and cols of subplots is dictated by no of patches used to formulated stacked_frc arr.
        i.e., height of the arr
    '''
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
        # then x-coording corresponds to frc-space
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
