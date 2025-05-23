import os
import pandas as pd
import numpy as np
import hickle as hkl
from preprocessing.indices import evi, bi, msavi2, grndvi
from preprocessing import interpolation
from preprocessing.whittaker_smoother import Smoother
from src.downloading.utils import calculate_and_save_best_images


min_all = [0.006576638437476157, 0.0162050812542916, 0.010040436408026246, 0.013351644159609368, 
               0.01965362020294499, 0.014229037918669413, 0.015289539940489814, 0.011993591210803388,
               0.008239871824216068, 0.006546120393682765, 0.0, 0.0, 0.0, -0.1409399364817101,
               -0.4973397113668104, -0.09731556326714398, -0.7193834232943873]
max_all = [0.2691233691920348, 0.3740291447318227, 0.5171435111009385, 0.6027466239414053,
               0.5650263218127718, 0.5747005416952773, 0.5933928435187305, 0.6034943160143434, 
               0.7472037842374304, 0.7000076295109483, 0.509269855802243, 0.948334642387533,
               0.6729257769285485, 0.8177635298774327, 0.35768999002433816, 0.7545951919107605, 0.7602693339366691]

def ndmi(arr):
    return (arr[..., 3] - arr[..., 8]) / (arr[..., 3] + arr[..., 8])


def smooth_large_tile(arr, dates, interp):

    """
    Deals with image normalization, smoothing, and regular 
    timestep interpolation for the entire array all at once
    """
    
    sm = Smoother(lmbd = 100, size = 24, nbands = arr.shape[-1], 
        dimx = arr.shape[1], dimy = arr.shape[2], outsize = 12)
    
    arr, dates, interp = deal_w_missing_px(arr, dates, interp)

    if arr.shape[-1] == 10:
        indices = make_and_smooth_indices(arr, dates)
    

    try:
       
        arr, max_distance = calculate_and_save_best_images(arr, dates)
  
    except:
        print("Skipping because of no images")
        arr = np.zeros((24, arr.shape[1], arr.shape[2], arr.shape[-1]), dtype = np.float32)
        dates = [0,]
   
    arr = sm.interpolate_array(arr)

    #indices = make_and_smooth_indices(arr, dates)
    


    if arr.shape[-1] == 10:
        out = np.zeros((arr.shape[0], arr.shape[1], arr.shape[2], 14), dtype = np.float32)
        out[..., :10] = arr 
        out[..., 10:] = indices
    else:
        out = arr


    return out, dates, interp

def make_indices(arr):
    indices = np.zeros(
        (arr.shape[0], arr.shape[1], arr.shape[2], 4), dtype = np.float32
    )
    indices[:, ..., 0] = evi(arr)
    indices[:, ...,  1] = bi(arr)
    indices[:, ...,  2] = msavi2(arr)
    indices[:, ...,  3] = grndvi(arr)
    return indices


def make_and_smooth_indices(arr, dates):
    """Calculates remote sensing indices
    (evi, bi, msavi2, grndvi) and smooths them
    with the Whittaker smoother
    """
    sm_indices = Smoother(lmbd = 100, 
                          size = 24, 
                          nbands = 4, 
                          dimx = arr.shape[1],
                          dimy = arr.shape[2], 
                          outsize = 12)

    indices = make_indices(arr)
    try:
        indices, _ = calculate_and_save_best_images(indices, dates)
    except:
        indices = np.zeros((24, arr.shape[1], arr.shape[2], 4), dtype = np.float32)
        dates = [0,]
    indices = sm_indices.interpolate_array(indices)
    return indices

def deal_w_missing_px(arr, dates, interp):
    missing_px = interpolation.id_missing_px(arr, 10)
    if len(missing_px) > 0:
        dates = np.delete(dates, missing_px)
        arr = np.delete(arr, missing_px, 0)
        interp = np.delete(interp, missing_px, 0)
        #print(f"Removing {len(missing_px)} missing images, leaving {len(dates)} / {len(dates)}")

    if np.sum(arr == 0) > 0:
        for i in range(arr.shape[0]):
            arr_i = arr[i]
            arr_i[arr_i == 0] = np.median(arr, axis = 0)[arr_i == 0]

    if np.sum(arr == 1) > 0:
        for i in range(arr.shape[0]):
            arr_i = arr[i]
            arr_i[arr_i == 1] = np.median(arr, axis = 0)[arr_i == 1]
    to_remove = np.argwhere(np.sum(np.isnan(arr), axis = (1, 2, 3)) > 0).flatten()
    if len(to_remove) > 0: 
        #print(f"Removing {to_remove} NA dates")
        dates = np.delete(dates, to_remove)
        arr = np.delete(arr, to_remove, 0)
        interp = np.delete(interp, to_remove, 0)
    return arr, dates, interp


def process_subtiles(local_path,x: int, y: int, s2: np.ndarray = None, 
                       dates: np.ndarray = None,
                       interp: np.ndarray = None, s1 = None, dem = None,
                       sess = None, bbx = None, size = 172-14, train_bbx = None,length=4) -> None:
    '''Wrapper function to interpolate clouds and temporal gaps, superresolve tiles,
       calculate relevant indices, and save predicted tree cover as a .npy
       
       Parameters:
        x (int): integer representation of the x tile ID
        y (int): integer representation of the y tile ID
        s2 (arr): (n, 160, 160, 11) array of sentinel 2 + DEM
        dates (arr): (n,) array of day of year of each image 
        interp (arr): (n, 160, 160) bool array of interpolated areas
        s1 (arr): (12, 160, 160, 2) float32 array of dB sentinel 1 data
        sess (tf.Session): tensorflow sesion to use for temporal predictions

       Returns:
        None
    '''
    x = str(int(x))
    y = str(int(y))
    x = x[:-2] if ".0" in x else x
    y = y[:-2] if ".0" in y else y
    
    s2 = interpolation.interpolate_na_vals(s2)
    s2 = np.float32(s2)
    
    s2_median = np.median(s2, axis = 0).astype(np.float32)
    s2_median = np.concatenate([s2_median, 
        np.median(evi(s2), axis = 0)[..., np.newaxis]], axis = -1)
    s2_median = np.concatenate([s2_median,
        np.median(bi(s2), axis = 0)[..., np.newaxis]], axis = -1)
    s2_median = np.concatenate([s2_median, 
        np.median(msavi2(s2), axis = 0)[..., np.newaxis]], axis = -1)
    s2_median = np.concatenate([s2_median, 
        np.median(grndvi(s2), axis = 0)[..., np.newaxis]], axis = -1)
    ard_ndmi_file = f"{local_path}{str(x)}/{str(y)}/ard_ndmi.hkl"
    ard_ndmi = (ndmi(s2) * 10000).astype(np.int16) // 5 * 5
    hkl.dump(ard_ndmi, ard_ndmi_file, mode='w', compression='gzip')
    #np.save(f"{args.local_path}{str(x)}/{str(y)}/ard_dates.npy", dates)

    #if WRITE_MONTHLY_TIFS:
        #for i in range(s2.shape[0]):
            #write_ard_to_tif(s2[i, ..., :3], bbx,
            #                 f"{str(x)}{str(y)}/{str(x)}X{str(y)}Y_{str(i)}",crs, "")
            #print(s2[i, ..., :1])
            #write_ard_to_tif(s2[i, ..., :10], bbx,
                             #f"{'/media/ate/2ca0e8f5-dc27-455a-8bac-b4bdefb367a7/temp/2022/'}{str(x)}X{str(y)}Y_{str(i)}",crs, "")

    s2, dates, interp = smooth_large_tile(s2, dates, interp)
    s2_median = s2_median[np.newaxis]
    #med_evi = np.percentile(s2_median[..., 10].flatten(), 0.5)
    s1_median = np.median(s1, axis = 0)[np.newaxis].astype(np.float32)
    #s2_median = np.median(s2, axis = 0)[np.newaxis].astype(np.float32)

    fname = f"{str(x)}X{str(y)}Y"
    #write_ard_to_tif(np.mean(s2[..., :3], axis = 0), bbx, fname,crs, "")
    key = f'/composite/{x}/{y}/{str(x)}X{str(y)}Y.tif'
    #uploader.upload(bucket = args.s3_bucket, key = key, file = fname + ".tif")
    dem_train = dem[np.newaxis, ..., np.newaxis]
    dem_train = dem_train.repeat(4, axis = 0)

    ###### MAR 22 #####
    path_to_tile = f'{local_path}/{str(x)}/{str(y)}/'
    if not os.path.exists(os.path.realpath(f"{path_to_tile}ard/")):
        os.makedirs(os.path.realpath(f"{path_to_tile}ard/"))
    s2med = np.mean(s2[..., :10], axis = 0)
    s1med = np.median(s1, axis = 0)
    ard_median = np.concatenate([s2med, dem[..., np.newaxis], s1med], axis = -1)
   
    hkl.dump(ard_median,
      f"{path_to_tile}ard/{str(x)}X{str(y)}Y_ard.hkl", 
      mode='w',
      compression='gzip')
    key = f'/ard/{x}/{y}/{str(x)}X{str(y)}Y_ard.hkl'
    #uploader.upload(bucket = args.s3_bucket, key = key, 
    #  file = f"{path_to_tile}ard/{str(x)}X{str(y)}Y_ard.hkl")
    #os.remove(f"{path_to_tile}ard/{str(x)}X{str(y)}Y_ard.hkl")
    ###### MAR 22 #####

    '''
    if args.gen_composite == True:
        composite_fname = f'{local_path}{str(x)}/{str(y)}/{str(x)}X{str(y)}Y_composite.hkl'
        composite = np.concatenate([s2_median, s1_median], axis = -1)
        composite = composite * 10000
        composite = np.clip(composite, -32000, 32000)
        composite = np.int16(composite)

        hkl.dump(composite,
                 composite_fname, 
                 mode='w',
                 compression='gzip')

        print(f"Saved composite to {composite_fname} of shape {composite.shape} and type {composite.dtype}")
    '''

    if 1 > 0:
        # At the equator, a tile is 618 height 618 width
        #print(PX_x)
        #if np.logical_and(PX_y == 0, PX_x == 0):
        #    PX_y = (s2.shape[1]) // 2
        #    PX_x = (s2.shape[2]) // 2
        #if int(x) > 0 and int(y) > 0:
        #print(f"Plot centroid: {PX_x, PX_y}")
        s2_train = s2 #[:, PX_y - 38:PX_y + 38, PX_x - 38:PX_x + 38, :]
        s1_train = s1 #[:, PX_y - 38:PX_y + 38, PX_x - 38:PX_x + 38, :]
        dem_train = dem #[PX_y - 38:PX_y + 38, PX_x - 38:PX_x + 38]
        print(f"Plot shape: {s2_train.shape}")
        #else:
        #    Px_y = (s2.shape[1]) // 2
        #    Px_x = (s2.shape[2]) // 2
        #    print(f"Plot centroid: {Px_x, Px_y}")
        #    s2_train = s2[:, Px_y - 38:Px_y + 38, Px_x - 38:Px_x + 38, :]
        #    s1_train = s1[:, Px_y - 38:Px_y + 38, Px_x - 38:Px_x + 38, :]
        #    dem_train = dem[Px_y - 38:Px_y + 38, Px_x - 38:Px_x + 38]
        #    print(f"Plot shape: {s2_train.shape}")

        dem_train = dem_train[np.newaxis, ..., np.newaxis]
        dem_train = dem_train.repeat(12, axis = 0)
        indices_mins = [-0.1409399364817101, -0.4973397113668104, 
            -0.09731556326714398, -0.7193834232943873]
        indices_maxes = [0.8177635298774327, 0.35768999002433816,
           0.7545951919107605, 0.7602693339366691]
        
        print("s2_train",s2_train.shape)
        print("dem",dem_train.shape)
        print("s1 train",s1_train.shape)
    

        train_sample = np.concatenate([s2_train[..., :10], 
            dem_train, 
            s1_train,
            s2_train[..., 10:]], axis = -1)
        for i in range(13, 17):
            train_sample[..., i] = np.clip(train_sample[..., i], min_all[i], max_all[i])
            train_sample[..., i] -= min_all[i]
        train_sample[..., -1] /= 2
        train_sample = np.clip(train_sample, 0, 1)
    
        print(  "train sample",train_sample.shape)
        #exit()   
   
    s2 = np.reshape(s2, (4, 3, s2.shape[1], s2.shape[2], s2.shape[3]))
    s2 = np.median(s2, axis = 1, overwrite_input = True)
    s1 = np.reshape(s1, (4, 3, s1.shape[1], s1.shape[2], s1.shape[3]))
    s1 = np.median(s1, axis = 1, overwrite_input = True)  

    # Handle the full tile prediction
    full_tile_all = np.zeros((length + 1, s2.shape[1], s2.shape[2], 17), dtype=np.float32)
    full_tile_all[:-1, ..., :10] = s2[..., :10]
    full_tile_all[:-1, ..., 11:13] = s1
    full_tile_all[:-1, ..., 13:] = s2[..., 10:]
    full_tile_all[:, ..., 10] = dem[np.newaxis, ..., np.newaxis].repeat(length + 1, axis=0).squeeze()
    full_tile_all[-1, ..., :10] = s2_median[..., :10]
    full_tile_all[-1, ..., 11:13] = s1_median
    full_tile_all[-1, ..., 13:] = s2_median[..., 10:]
    #print("full sample",full_tile_all.shape)
    #exit()
    #    np.save(f"/home/ate/temp/{y}_{str(x)}_tile.npy", full_tile_all)    
    return full_tile_all
    

def process_subtiles_label(local_path,x: int, y: int, s2: np.ndarray = None, 
                       dates: np.ndarray = None,
                       interp: np.ndarray = None, s1 = None, dem = None,
                       sess = None, bbx = None, size = 172-14, train_bbx = None,length=4) -> None:
    '''Wrapper function to interpolate clouds and temporal gaps, superresolve tiles,
       calculate relevant indices, and save predicted tree cover as a .npy
       
       Parameters:
        x (int): integer representation of the x tile ID
        y (int): integer representation of the y tile ID
        s2 (arr): (n, 160, 160, 11) array of sentinel 2 + DEM
        dates (arr): (n,) array of day of year of each image 
        interp (arr): (n, 160, 160) bool array of interpolated areas
        s1 (arr): (12, 160, 160, 2) float32 array of dB sentinel 1 data
        sess (tf.Session): tensorflow sesion to use for temporal predictions

       Returns:
        None
    '''
    x = str(int(x))
    y = str(int(y))
    x = x[:-2] if ".0" in x else x
    y = y[:-2] if ".0" in y else y
    
    s2 = interpolation.interpolate_na_vals(s2)
    s2 = np.float32(s2)
    
    s2_median = np.median(s2, axis = 0).astype(np.float32)
    s2_median = np.concatenate([s2_median, 
        np.median(evi(s2), axis = 0)[..., np.newaxis]], axis = -1)
    s2_median = np.concatenate([s2_median,
        np.median(bi(s2), axis = 0)[..., np.newaxis]], axis = -1)
    s2_median = np.concatenate([s2_median, 
        np.median(msavi2(s2), axis = 0)[..., np.newaxis]], axis = -1)
    s2_median = np.concatenate([s2_median, 
        np.median(grndvi(s2), axis = 0)[..., np.newaxis]], axis = -1)
    ard_ndmi_file = f"{local_path}{str(x)}/{str(y)}/ard_ndmi.hkl"
    ard_ndmi = (ndmi(s2) * 10000).astype(np.int16) // 5 * 5
    hkl.dump(ard_ndmi, ard_ndmi_file, mode='w', compression='gzip')
    #np.save(f"{args.local_path}{str(x)}/{str(y)}/ard_dates.npy", dates)

    #if WRITE_MONTHLY_TIFS:
        #for i in range(s2.shape[0]):
            #write_ard_to_tif(s2[i, ..., :3], bbx,
            #                 f"{str(x)}{str(y)}/{str(x)}X{str(y)}Y_{str(i)}",crs, "")
            #print(s2[i, ..., :1])
            #write_ard_to_tif(s2[i, ..., :10], bbx,
                             #f"{'/media/ate/2ca0e8f5-dc27-455a-8bac-b4bdefb367a7/temp/2022/'}{str(x)}X{str(y)}Y_{str(i)}",crs, "")

    s2, dates, interp = smooth_large_tile(s2, dates, interp)
    s2_median = s2_median[np.newaxis]
    #med_evi = np.percentile(s2_median[..., 10].flatten(), 0.5)
    s1_median = np.median(s1, axis = 0)[np.newaxis].astype(np.float32)
    #s2_median = np.median(s2, axis = 0)[np.newaxis].astype(np.float32)

    fname = f"{str(x)}X{str(y)}Y"
    #write_ard_to_tif(np.mean(s2[..., :3], axis = 0), bbx, fname,crs, "")
    key = f'/composite/{x}/{y}/{str(x)}X{str(y)}Y.tif'
    #uploader.upload(bucket = args.s3_bucket, key = key, file = fname + ".tif")
    dem_train = dem[np.newaxis, ..., np.newaxis]
    dem_train = dem_train.repeat(4, axis = 0)

    ###### MAR 22 #####
    path_to_tile = f'{local_path}/{str(x)}/{str(y)}/'
    if not os.path.exists(os.path.realpath(f"{path_to_tile}ard/")):
        os.makedirs(os.path.realpath(f"{path_to_tile}ard/"))
    s2med = np.mean(s2[..., :10], axis = 0)
    s1med = np.median(s1, axis = 0)
    ard_median = np.concatenate([s2med, dem[..., np.newaxis], s1med], axis = -1)
   
    hkl.dump(ard_median,
      f"{path_to_tile}ard/{str(x)}X{str(y)}Y_ard.hkl", 
      mode='w',
      compression='gzip')
    key = f'/ard/{x}/{y}/{str(x)}X{str(y)}Y_ard.hkl'
    #uploader.upload(bucket = args.s3_bucket, key = key, 
    #  file = f"{path_to_tile}ard/{str(x)}X{str(y)}Y_ard.hkl")
    #os.remove(f"{path_to_tile}ard/{str(x)}X{str(y)}Y_ard.hkl")
    ###### MAR 22 #####

    '''
    if args.gen_composite == True:
        composite_fname = f'{local_path}{str(x)}/{str(y)}/{str(x)}X{str(y)}Y_composite.hkl'
        composite = np.concatenate([s2_median, s1_median], axis = -1)
        composite = composite * 10000
        composite = np.clip(composite, -32000, 32000)
        composite = np.int16(composite)

        hkl.dump(composite,
                 composite_fname, 
                 mode='w',
                 compression='gzip')

        print(f"Saved composite to {composite_fname} of shape {composite.shape} and type {composite.dtype}")
    '''

    s2_train = s2 #[:, PX_y - 38:PX_y + 38, PX_x - 38:PX_x + 38, :]
    s1_train = s1 #[:, PX_y - 38:PX_y + 38, PX_x - 38:PX_x + 38, :]
    dem_train = dem #[PX_y - 38:PX_y + 38, PX_x - 38:PX_x + 38]
    
    dem_train = dem_train[np.newaxis, ..., np.newaxis]
    dem_train = dem_train.repeat(12, axis = 0)    

    train_sample = np.concatenate([s2_train[..., :10], 
        dem_train, 
        s1_train,
        s2_train[..., 10:]], axis = -1)

    
    print(  "train sample",train_sample.shape)

    return train_sample
