import numpy as np

def psnr(f_true, f_est, max_val=1.0):
    
    imdff = f_true - f_est
    rmse = np.sqrt(np.mean(imdff **2))
    psnr = 20.0*np.log10(max_val/rmse)
    return(psnr)

def relative_mse(f_true, f_est):
    imdiff = f_true-f_est
    nume = np.sum(imdiff**2)
    deno = np.mean(f_true)-f_true
    deno = np.sum(deno**2)
    return(nume/deno)
