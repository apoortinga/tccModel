import pandas as pd
import numpy as np
import hickle as hkl
import os
import time
from skimage.transform import resize
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import median_filter, maximum_filter, percentile_filter

from preprocessing import interpolation
from preprocessing import cloud_removal

WRITE_RAW_TIFS = False

def to_float32(array: np.array) -> np.array:
    """Converts an int array to float32"""
    #print(f'The original max value is {np.max(array)}')
    if not isinstance(array.flat[0], np.floating):
        assert np.max(array) > 1
        array = np.divide(np.float32(array), 65535.)
    assert np.max(array) <= 1
    assert array.dtype == np.float32
    return array


def convert_to_db(x: np.ndarray, min_db: int) -> np.ndarray:
    """ Converts Sentinel 1 unitless backscatter coefficient
        to db with a min_db lower threshold
        
        Parameters:
         x (np.ndarray): unitless backscatter (T, X, Y, B) array
         min_db (int): integer from -50 to 0
    
        Returns:
         x (np.ndarray): db backscatter (T, X, Y, B) array
    """
    
    x = 10 * np.log10(x + 1/65535)
    x[x < -min_db] = -min_db
    x = (x + min_db) / min_db
    return np.clip(x, 0, 1)


def adjust_shape(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Assures that the shape of arr is width x height
    Used to align 10, 20
    0, 160, 640 meter resolution Sentinel data
    """
    #print(f"Input array shape: {arr.shape}")
    arr = arr[:, :, :, np.newaxis] if len(arr.shape) == 3 else arr
    arr = arr[np.newaxis, :, :, np.newaxis] if len(arr.shape) == 2 else arr
    
    if arr.shape[1] < width:
        pad_amt = (width - arr.shape[1]) // 2
        if pad_amt == 0:
            arr = np.pad(arr, ((0, 0), (1, pad_amt), (0,0), (0, 0)), 'edge')
        else:
            arr = np.pad(arr, ((0, 0), (pad_amt, pad_amt), (0,0), (0, 0)), 'edge')

    if arr.shape[2] < height:
        pad_amt = (height - arr.shape[2]) // 2
        if pad_amt == 0:
            arr = np.pad(arr, ((0, 0), (0,0), (1, 0), (0, 0)), 'edge')
        else:
            arr = np.pad(arr, ((0, 0), (0,0), (pad_amt, pad_amt), (0, 0)), 'edge')

    if arr.shape[1] > width:
        pad_amt =  (arr.shape[1] - width) // 2
        pad_amt_even = (arr.shape[1] - width) % 2 == 0
        if pad_amt == 0:
            arr = arr[:, 1:, ...]
        elif pad_amt_even:
            pad_amt = int(pad_amt)
            arr = arr[:, pad_amt:-pad_amt, ...]
        else:
            pad_left = int(np.floor(pad_amt / 2))
            pad_right = int(np.ceil(pad_amt / 2))
            arr = arr[:, pad_left:-pad_right, ...]

    if arr.shape[2] > height:
        pad_amt = (arr.shape[2] - height) // 2
        pad_amt_even = (arr.shape[2] - height) % 2 == 0
        if pad_amt == 0:
            arr = arr[:, :, 1:, :]
        elif pad_amt_even:
            pad_amt = int(pad_amt)
            arr = arr[:, :, pad_amt:-pad_amt, ...]
        else:
            pad_left = int(np.floor(pad_amt / 2))
            pad_right = int(np.ceil(pad_amt / 2))
            arr = arr[:, :, pad_left:-pad_right, ...]

    return arr.squeeze()

def process_tile(x: int, y: int, 
                 local_path: str, bbx, make_shadow: bool = False) -> np.ndarray:
    """
    Processes raw data structure (in temp/raw/*) to processed data structure
        - align shapes of different data sources (clouds / shadows / s1 / s2 / dem)
        - superresolve 20m to 10m with bilinear upsampling for DSen2 input
        - remove (interpolate) clouds and shadows

    Parameters:
         x (int): x position of tile to be downloaded
         y (int): y position of tile to be downloaded
         data (pd.DataFrame): tile grid dataframe

        Returns:
         x (np.ndarray)
         image_dates (np.ndarray)
         interp (np.ndarray)
         s1 (np.ndarray)
    """
    
    def _water_ndwi(array):
        return (array[..., 1] - array[..., 3]) / (array[..., 1] + array[..., 3])

    x = str(int(x))
    y = str(int(y))
    x = x[:-2] if ".0" in x else x
    y = y[:-2] if ".0" in y else y
    
    #print("bbx",bbx)
            
    folder = f"{local_path}{str(x)}/{str(y)}/"
    tile_idx = f'{str(x)}X{str(y)}Y'
    
    clouds_file = f'{folder}raw/clouds/clouds_{tile_idx}.hkl'
    cloud_mask_file = f'{folder}raw/clouds/cloudmask_{tile_idx}.hkl'
    shadows_file = f'{folder}raw/clouds/shadows_{tile_idx}.hkl'
    s1_file = f'{folder}raw/s1/{tile_idx}.hkl'
    s1_dates_file = f'{folder}raw/misc/s1_dates_{tile_idx}.hkl'
    s2_10_file = f'{folder}raw/s2_10/{tile_idx}.hkl'
    s2_20_file = f'{folder}raw/s2_20/{tile_idx}.hkl'
    s2_dates_file = f'{folder}raw/misc/s2_dates_{tile_idx}.hkl'
    s2_file = f'{folder}raw/s2/{tile_idx}.hkl'
    clean_steps_file = f'{folder}raw/clouds/clean_steps_{tile_idx}.hkl'
    dem_file = f'{folder}raw/misc/dem_{tile_idx}.hkl'
    
    clouds = hkl.load(clouds_file)

    if os.path.exists(cloud_mask_file):
        # These are the S2Cloudless / Sen2Cor masks
        clm = hkl.load(cloud_mask_file).repeat(2, axis = 1).repeat(2, axis = 2)
        for i in range(0, clm.shape[0]):
            mins = np.maximum(i - 1, 0)
            maxs = np.minimum(i + 1, clm.shape[0])
            # if 2 in a row are clouds, remove
            # since the Sen2Cor mask has high FP
            sums = np.sum(clm[mins:maxs], axis = 0) == 2
            clm[mins:maxs, sums] = 0.
        #print("The Sen2Cor cloud percentages are: ", np.mean(clm, axis = (1, 2)))
    else:
        clm = None

    s1 = hkl.load(s1_file)
    s1 = np.float32(s1) / 65535
    #print(s1.shape)
    for i in range(s1.shape[0]):
        s1_i = s1[i]
        s1_i[s1_i == 1] = np.median(s1_i[s1_i < 65535], axis = 0)
        s1[i] = s1_i

    s1[..., -1] = convert_to_db(s1[..., -1], 22)
    s1[..., -2] = convert_to_db(s1[..., -2], 22)
    s1 = s1.astype(np.float32)

    s2_10 = to_float32(hkl.load(s2_10_file))   
    s2_20 = to_float32(hkl.load(s2_20_file))
    dem = hkl.load(dem_file)
    dem = median_filter(dem, size =5)
    image_dates = hkl.load(s2_dates_file)
    
    # Ensure arrays are the same dims
    width = s2_20.shape[1] * 2
    height = s2_20.shape[2] * 2
    s1 = adjust_shape(s1, width, height)
    s2_10 = adjust_shape(s2_10, width, height)
    dem = adjust_shape(dem, width, height)

    #print(f'### Array shapes ### \nClouds: {clouds.shape}, \n'
    #      f'S1: {s1.shape} \n'
    #      f'S2: {s2_10.shape}, {s2_20.shape} \n'
    #      f'DEM: {dem.shape}')

    # Deal with cases w/ only 1 image
    if len(s2_10.shape) == 3:
        s2_10 = s2_10[np.newaxis]
    if len(s2_20.shape) == 3:
        s2_20 = s2_20[np.newaxis]

    # bilinearly upsample 20m bands to 10m for superresolution
    sentinel2 = np.zeros((s2_10.shape[0], width, height, 10), np.float32)
    sentinel2[..., :4] = s2_10

    # a foor loop is faster than trying to vectorize it here! 
    for band in range(4):
        for step in range(sentinel2.shape[0]):
            sentinel2[step, ..., band + 4] = resize(
                s2_20[step,..., band], (width, height), 1
            )
    for band in range(4, 6):
        # indices 4, 5 are 40m and may be a different shape
        # this code is ugly, but it forces the arrays to match up w/ the 10/20m ones
        for step in range(sentinel2.shape[0]):
            mid = s2_20[step,..., band]
            #if args.make_training_data == True:
            #    print('wtf')
                #mid = mid.reshape(mid.shape[0] // 2, 2, mid.shape[1] // 2, 2)
                #mid = np.mean(mid, axis = (1, 3))
                #sentinel2[step, ..., band + 4] = resize(mid, (width, height), 1)
            if (mid.shape[0] % 2 == 0) and (mid.shape[1] % 2) == 0:
                # So bands 4, 5 need to be bilinearly upsampled for the input to
                # The super-resolution
                mid = mid.reshape(mid.shape[0] // 2, 2, mid.shape[1] // 2, 2)
                mid = np.mean(mid, axis = (1, 3))
                sentinel2[step, ..., band + 4] = resize(mid, (width, height), 1)
            elif mid.shape[0] %2 != 0 and mid.shape[1] %2 != 0:
                mid_misaligned_x = mid[0, :]
                mid_misaligned_y = mid[:, 0]
                mid = mid[1:, 1:].reshape(
                    np.int32(np.floor(mid.shape[0] / 2)), 2,
                    np.int32(np.floor(mid.shape[1] / 2)), 2)
                mid = np.mean(mid, axis = (1, 3))
                sentinel2[step, 1:, 1:, band + 4] = resize(mid, (width - 1, height - 1), 1)
                sentinel2[step, 0, :, band + 4] = mid_misaligned_x.repeat(2)
                sentinel2[step, :, 0, band + 4] = mid_misaligned_y.repeat(2)
            elif mid.shape[0] % 2 != 0:
                mid_misaligned = mid[0, :]
                mid = mid[1:].reshape(np.int32(np.floor(mid.shape[0] / 2)), 2, mid.shape[1] // 2, 2)
                mid = np.mean(mid, axis = (1, 3))
                sentinel2[step, 1:, :, band + 4] = resize(mid, (width - 1, height), 1)
                sentinel2[step, 0, :, band + 4] = mid_misaligned.repeat(2)
            elif mid.shape[1] % 2 != 0:
                mid_misaligned = mid[:, 0]
                mid = mid[:, 1:]
                mid = mid.reshape(mid.shape[0] // 2, 2, np.int32(np.floor(mid.shape[1] / 2)), 2)
                mid = np.mean(mid, axis = (1, 3))
                sentinel2[step, :, 1:, band + 4] = resize(mid, (width, height - 1), 1)
                sentinel2[step, :, 0, band + 4] = mid_misaligned.repeat(2)
    #print(f"SENTINEL2, {sentinel2.shape}")
    # Identifies missing imagery (either in sentinel acquisition, or induced in preprocessing)
    # If more than 50% of data for a time step is missing, then remove them....
    missing_px = interpolation.id_missing_px(sentinel2, 2)
    if len(missing_px) > 0:
        #print(f"Removing {missing_px} dates due to {missing_px} missing data")
        if clouds.shape[0] == len(image_dates):
            clouds = np.delete(clouds, missing_px, axis = 0)
        image_dates = np.delete(image_dates, missing_px)
        sentinel2 = np.delete(sentinel2, missing_px, axis = 0)
        if clm is not None:
            clm = np.delete(clm, missing_px, axis = 0)


    # Remove images that are >10% snow...
    def ndsi(arr):
        return (arr[..., 1] - arr[..., 8]) / (arr[..., 1] + arr[..., 8])

    def snow_filter(arr):
        ndsi =  (arr[..., 1] - arr[..., 8]) / (arr[..., 1] + arr[..., 8])
        ndsi[ndsi < 0.10] = 0.
        ndsi[ndsi > 0.42] = 0.42
        snow_prob = (ndsi - 0.1) / 0.32

        # NIR band threshold
        snow_prob[arr[..., 3] < 0.10] = 0.
        snow_prob[np.logical_and(arr[..., 3] > 0.35, snow_prob > 0)] = 1.

        # blue band threshold
        snow_prob[arr[..., 0] < 0.10] = 0.
        snow_prob[np.logical_and(arr[..., 0] > 0.22, snow_prob > 0)] = 1.

        # B2/B4 thrershold
        b2b4ratio = arr[..., 0] / arr[..., 2]
        snow_prob[b2b4ratio < 0.75] = 0.
        return snow_prob > 0

    ndsis = snow_filter(sentinel2)
    mean_snow_per_img = np.mean(ndsis, axis = (1, 2))
    snow = np.mean(ndsis, axis = 0)
    snow = 1 - binary_dilation(snow < 0.7, iterations = 2)
    to_remove = np.argwhere(mean_snow_per_img > 0.25).flatten()
    # CURRENTLY DEFUNCT ## 
    if (len(to_remove) > 10):# and args.snow:
        #print(f"Removing {to_remove} dates due to {to_remove} snow cover")
        if clouds.shape[0] == len(image_dates):
            clouds = np.delete(clouds, to_remove, axis = 0)
        #print("image shape",image_dates.shape)
        #print("to remove shape",to_remove.shape)
        image_dates = np.delete(image_dates, to_remove)
        sentinel2 = np.delete(sentinel2, to_remove, axis = 0)
        if clm is not None:
            clm = np.delete(clm, to_remove, axis = 0)
    
    # Otherwise... set the missing values to the median value.
    sentinel2 = interpolation.interpolate_missing_vals(sentinel2)
    
    if make_shadow:
        time1 = time.time()
        # Bounding box passed to identify_cloud_shadows to mask 
        # out non-urban areas from the false positive cloud removal
        #! TODO: https://www.sciencedirect.com/science/article/pii/S0034425718302037
        cloudshad, fcps = cloud_removal.identify_clouds_shadows(sentinel2, dem, bbx)
        if clm is not None:
            try:
                clm[fcps] = 0.
                cloudshad = np.maximum(cloudshad, clm)
            except:
                print("Error, continuing")

        interp = cloud_removal.id_areas_to_interp(
            sentinel2, cloudshad, cloudshad, image_dates, pfcps = fcps
        )
        #print(f"IMAGE DATES: {image_dates}")
        interp_pct = 100*np.mean(interp >0.05, axis = (1, 2))
        #print(f"Interpolate per-image: {np.around(interp_pct, 1)}")
        # In order to properly normalize band values to gapfill cloudy areas
        # We need 10% of the image to be non-cloudy
        # So that we can identify PIFs with at least 1000 px
        # Images deleted here will get propogated in the resegmentation
        # So it should not cause boundary artifacts in the final product.
        
        water_mask = _water_ndwi(np.median(sentinel2, axis=0)) > 0.0
        means = np.mean(interp == 1, axis = (1, 2))
        to_remove = np.argwhere(np.mean(interp > 0, axis = (1, 2)) > 0.9).flatten() 
        if len(to_remove) > 0:
            #print(f"clouds 1{clouds.shape}")
            #print(f"Deleting {to_remove}")
            #print(clouds.shape, image_dates.shape, sentinel2.shape, interp.shape,to_remove.shape)
            if clouds.shape[0] == len(image_dates):
                clouds = np.delete(clouds, to_remove, axis = 0)
            #print(clouds.shape, image_dates.shape, sentinel2.shape, interp.shape,to_remove.shape)
            
            image_dates = np.delete(image_dates, to_remove)
            sentinel2 = np.delete(sentinel2, to_remove, axis = 0)
            interp = np.delete(interp, to_remove, axis = 0)
            cloudshad, fcps = cloud_removal.identify_clouds_shadows(sentinel2, dem, bbx)
            if clm is not None:
                clm = np.delete(clm, to_remove, axis = 0)
                try:
                    cloudshad = np.maximum(cloudshad, clm)
                except:
                    print("CLM error, continuing")

            interp = cloud_removal.id_areas_to_interp(
                sentinel2, cloudshad, cloudshad, image_dates, pfcps = fcps
            )
        to_remove = np.argwhere(np.mean(interp > 0, axis = (1, 2)) > 0.9).flatten()
        if len(to_remove) > 0:
            #print(f"clouds 2{clouds.shape}")
            #print(f"Deleting {to_remove}")
            #print(clouds.shape, image_dates.shape, sentinel2.shape, interp.shape)
            if clouds.shape[0] == len(image_dates):
                clouds = np.delete(clouds, to_remove, axis = 0)
            image_dates = np.delete(image_dates, to_remove)
            sentinel2 = np.delete(sentinel2, to_remove, axis = 0)
            interp = np.delete(interp, to_remove, axis = 0)
            cloudshad, fcps = cloud_removal.identify_clouds_shadows(sentinel2, dem, bbx)
            if clm is not None:
                clm = np.delete(clm, to_remove, axis = 0)
                try:
                    cloudshad = np.maximum(cloudshad, clm)
                except:
                    print("CLM error, continuing")

            interp = cloud_removal.id_areas_to_interp(
                sentinel2, cloudshad, cloudshad, image_dates, pfcps = fcps
            )
        to_remove = np.argwhere(np.mean(interp > 0, axis = (1, 2)) > 0.9).flatten()
        if len(to_remove) > 0:
            #print(f"Deleting {to_remove}")
            clouds = np.delete(clouds, to_remove, axis = 0)
            image_dates = np.delete(image_dates, to_remove)
            sentinel2 = np.delete(sentinel2, to_remove, axis = 0)
            interp = np.delete(interp, to_remove, axis = 0)
            cloudshad, fcps = cloud_removal.identify_clouds_shadows(sentinel2, dem, bbx)
            if clm is not None:
                clm = np.delete(clm, to_remove, axis = 0)
                try:
                    cloudshad = np.maximum(cloudshad, clm)
                except:
                    print("CLM error, continuing")
        interp = cloud_removal.id_areas_to_interp(
                sentinel2, cloudshad, cloudshad, image_dates, pfcps = fcps
            )
        

        np.save('interp.npy', interp)

        def _ndwi(arr):
            return (arr[..., 1] - arr[..., 3]) / (arr[..., 1] + arr[..., 3])

        water_mask = np.median(_ndwi(sentinel2), axis=0)
        water_mask = water_mask > 0
        def _ndbi(arr):
            return ((arr[..., 8] - arr[..., 3]) / (arr[..., 8] + arr[..., 3]))
        ndbi = np.median(_ndbi(sentinel2), axis = 0)

        if WRITE_RAW_TIFS:
            for i in range(sentinel2.shape[0]):
                write_ard_to_tif(sentinel2[i, ..., :3], bbx,
                                f"{str(x)}{str(y)}/{str(x)}X{str(y)}Y_{str(i)}_RAW",crs, "")
        _, interp, to_remove = cloud_removal.remove_cloud_and_shadows(
                sentinel2, cloudshad, cloudshad, image_dates,
                 pfcps = fcps, 
                 sentinel1 = #np.mean(s1, axis = 0),
                 np.concatenate([np.mean(s1, axis = 0),
                  dem[..., np.newaxis],
                  water_mask[..., np.newaxis],
                  ndbi[..., np.newaxis]], axis = -1),
                mosaic = None,
            )
        #write_ard_to_tif(mosaic[..., :3], bbx,
        #                        f"{str(x)}{str(y)}/{str(x)}X{str(y)}Y_MOSAIC", "")
        interp_pct = 100*np.mean(interp >0.05, axis = (1, 2))
        #print(f"Interpolate per-image: {np.around(interp_pct, 1)}")

        # If the brightness increases by a lot, and the std decreases by a lot
        # Then we're calling that haze.
        mean_brightness_per_img = np.mean(sentinel2[..., :3], axis = -1)
        mean_brightness = np.mean(mean_brightness_per_img, axis = (1, 2))
        std_brightness = np.std(mean_brightness_per_img, axis = (1, 2))
        #print("B", mean_brightness, std_brightness)
        is_haze = np.diff(mean_brightness) > (np.mean(mean_brightness) * 0.5)
        is_haze = is_haze * np.diff(std_brightness) < (np.mean(std_brightness) * -0.5)
        is_haze = np.argwhere(is_haze > 0)
        #if len(is_haze) > 0:
       #     is_haze = is_haze + 1
        #    is_haze = is_haze.flatten()
        #    to_remove = to_remove + list(is_haze)
        #print(f"HAZE FLAG: {is_haze}")
        cloudlength  = clouds.shape[0]
        to_remove = [item for item in to_remove if item < cloudlength]
        if len(to_remove) > 0:
            #print(f"clouds {clouds.shape}")
            #print(f"Deleting {to_remove}")

            clouds = np.delete(clouds, to_remove, axis = 0)
            image_dates = np.delete(image_dates, to_remove)
            sentinel2 = np.delete(sentinel2, to_remove, axis = 0)

            interp = np.delete(interp, to_remove, axis = 0)
            cloudshad, fcps = cloud_removal.identify_clouds_shadows(sentinel2, dem, bbx)
            
            if clm is not None:
                clm = np.delete(clm, to_remove, axis = 0)
                try:
                    cloudshad = np.maximum(cloudshad, clm)
                except:
                    print("CLM error, continuing")

            interp = cloud_removal.id_areas_to_interp(
                sentinel2, cloudshad, cloudshad, image_dates, pfcps = fcps
            )
        interp_pct = 100*np.mean(interp >0.05, axis = (1, 2))
        #print(f"Interpolate per-image: {np.around(interp_pct, 1)}")

    else:
        interp = np.zeros(
            (sentinel2.shape[0], sentinel2.shape[1], sentinel2.shape[2]), dtype = np.float32
        )
        cloudshad = np.zeros(
            (sentinel2.shape[0], sentinel2.shape[1], sentinel2.shape[2]), dtype = np.float32
        )

    dem = dem / 90
    sentinel2 = np.clip(sentinel2, 0, 1)
    return sentinel2, image_dates, interp, s1, dem, cloudshad, snow
