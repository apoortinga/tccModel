import yaml
import pandas as pd
import copy
import random
import os
import numpy as np
import hickle as hkl
import tensorflow as tf
from tqdm import tqdm
import rasterio
from rasterio.transform import from_bounds
import argparse
from google.cloud import storage


if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)



from tof import gee_downloading
from downloading.io import make_output_and_temp_folders
from preprocessing import cloud_removal
from preprocessing import interpolation
from preprocessing.whittaker_smoother import Smoother
from processing.process import process_tile
from tof.tof_downloading import to_int16, to_float32
from processing.subtile import process_subtiles,process_subtiles_label
from ee_auth import ee_initialize

ee_initialize()

SIZE = 172-14

'''
def upload_to_gcs(local_file_path, bucket_name, blob_path,year):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_file_path)
    print(f"Uploaded {local_file_path} to gs://{bucket_name}/{blob_path}/{year}")
'''

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_bbox(initial_bbx: list, expansion: int = 10) -> list:
    """Makes a (min_x, min_y, max_x, max_y) bounding box that
       is 2 * expansion 300 x 300 meter ESA LULC pixels
       e.g. expansion = 10 generates a 6 x 6 km tile
       Parameters:
            initial_bbx (list): [min_x, min_y, max_x, max_y]
            expansion (int): 1/2 number of 300m pixels to expand

       Returns:
            bbx (list): expanded [min_x, min_y, max_x, max_y]
    """
    multiplier = 1/360 # Sentinel-2 pixel size in decimal degrees
    bbx = copy.deepcopy(initial_bbx)
    bbx[0] -= expansion * multiplier
    bbx[1] -= expansion * multiplier
    bbx[2] += expansion * multiplier
    bbx[3] += expansion * multiplier
    return bbx

def generate_date_range(year):
    start_date = f'{year - 1}-11-15'
    end_date = f'{year + 1}-02-15'
    return start_date, end_date


def download_s1_tile_gee(bbox,year,s1_file,s1_dates_file,crs):
    
    
    s1, s1_dates = gee_downloading.download_sentinel_1_composite(bbox,  year,crs = crs)

    hkl.dump(to_int16(s1), s1_file, mode='w', compression='gzip')
    hkl.dump(s1_dates, s1_dates_file, mode='w', compression='gzip')




def superresolve_large_tile(arr: np.ndarray, sess) -> np.ndarray:
    """Superresolves an input tile utilizing the open tf.sess().
       Implements a lightweight version of DSen2, a CNN-based
       image superresolution model

       Reference: https://arxiv.org/abs/1803.04271

       Parameters:
            arr (arr): (?, X, Y, 10) array, where arr[..., 4:]
                       has been bilinearly upsampled

       Returns:
            superresolved (arr): (?, X, Y, 10) array
    """
    # Pad the input images to avoid border artifacts
    def _worker_fn(arr: np.ndarray, sess) -> np.ndarray:
        # Pad the input images to avoid border artifacts
        to_resolve = np.pad(arr, ((0, 0), (4, 4), (4, 4), (0, 0)), 'reflect')

        bilinear = to_resolve[..., 4:]
        resolved = sess.run([superresolve_logits], 
                     feed_dict={superresolve_inp: to_resolve,
                                superresolve_inp_bilinear: bilinear})[0]
        resolved = resolved[:, 4:-4, 4:-4, :]
        arr[..., 4:] = resolved
        return arr

    wsize = 110
    step = 110
    x_range = [x for x in range(0, arr.shape[1] - (wsize), step)] + [arr.shape[1] - wsize]
    y_range = [x for x in range(0, arr.shape[2] - (wsize), step)] + [arr.shape[2] - wsize]
    x_end = np.copy(arr[:, x_range[-1]:, ...])
    y_end = np.copy(arr[:, :, y_range[-1]:, ...])

    
    for x in tqdm(x_range):
        for y in y_range:
            if x != x_range[-1] and y != y_range[-1]:
                to_resolve = arr[:, x:x+wsize, y:y+wsize, ...]
                arr[:, x:x+wsize, y:y+wsize, ...] = _worker_fn(to_resolve, sess)
            # The end x and y subtiles need to be done separately
            # So that a partially resolved tile isnt served as input
            elif x == x_range[-1]:
                to_resolve = x_end[:, :, y:y+wsize, ...]
                arr[:, x:x+wsize, y:y+wsize, ...] = _worker_fn(to_resolve, sess)
            elif y != y_range[-1]:
                to_resolve = y_end[:, x:x+wsize, :, ...]
                arr[:, x:x+wsize, y:y+wsize, ...] = _worker_fn(to_resolve, sess)


    return arr

def download_tile(x: int, y: int,  year, initial_bbx, expansion, local_path = "/tmp/app/") -> None:
    """Downloads the data for an input x, y tile centroid
       including:
        - Clouds
        - Sentinel 1
        - Sentinel 2 (10 and 20 m)
        - DEM

       Writes the raw data to the output/x/y folder as .hkl structure

       Parameters:
            x (int): x position of tile to be downloaded
            y (int): y position of tile to be downloaded
            data (pd.DataFrame): tile grid dataframe

       Returns:
            None

    """
    
    # Are selected in neighboring tiles
    cloud_bbx = make_bbox(initial_bbx, expansion = (expansion * 15)/30)
    bbx = make_bbox(initial_bbx, expansion = expansion/30)
    dem_bbx = make_bbox(initial_bbx, expansion = (expansion + 1)/30)
   
    crs = gee_downloading.getCRS(initial_bbx)

   
    os.makedirs(local_path, exist_ok=True)

    folder = f"{local_path}/{str(x)}/{str(y)}/"
    tile_idx = f'{str(x)}X{str(y)}Y'
    print("exporting data to ",folder)
    
    make_output_and_temp_folders(folder)

    clouds_file = f'{folder}raw/clouds/clouds_{tile_idx}.hkl'
    cloud_mask_file = f'{folder}raw/clouds/cloudmask_{tile_idx}.hkl'
    shadows_file = f'{folder}raw/clouds/shadows_{tile_idx}.hkl'
    s1_file = f'{folder}raw/s1/{tile_idx}.hkl'
    s1_dates_file = f'{folder}raw/misc/s1_dates_{tile_idx}.hkl'
    s2_10_file = f'{folder}raw/s2_10/{tile_idx}.hkl'
    s2_20_file = f'{folder}raw/s2_20/{tile_idx}.hkl'
    s2_dates_file = f'{folder}raw/misc/s2_dates_{tile_idx}.hkl'
    s2_file = f'{folder}raw/s2/{tile_idx}.hkl' # deprecated?
    clean_steps_file = f'{folder}raw/clouds/clean_steps_{tile_idx}.hkl'
    clean_filename = f'{folder}raw/clouds/clean_fname_{tile_idx}.hkl'
    dem_file = f'{folder}raw/misc/dem_{tile_idx}.hkl'

    dates = generate_date_range(year)
    
    if (os.path.exists(clouds_file)):
        myBox = gee_downloading.get_bbx(cloud_bbx = initial_bbx,crs = crs)
    
    if not (os.path.exists(clouds_file)):
        print(f"starting downloading..")
      

        # Identify images with <30% cloud cover
        cloud_probs, cloud_percent, all_dates, all_local_clouds, filenames, myBox = gee_downloading.identify_clouds_big_bbx(
            cloud_bbx = initial_bbx, 
            dates = dates,
            crs = crs,
            year = year
        )
                
        cloud_probs = cloud_probs * 100
        cloud_probs[cloud_probs > 100] = np.nan
        cloud_percent = np.nanmean(cloud_probs, axis = (1, 2))
        cloud_percent = cloud_percent / 100

        local_clouds = np.copy(all_local_clouds)
        image_dates = np.copy(all_dates)

        to_remove = cloud_removal.subset_contiguous_sunny_dates(image_dates, cloud_percent)
        
        if len(to_remove) > 0:
            clean_dates = np.delete(image_dates, to_remove)
            clean_filenames = np.delete(filenames, to_remove)
            cloud_probs = np.delete(cloud_probs, to_remove, 0)
            cloud_percent = np.delete(cloud_percent, to_remove)
            local_clouds = np.delete(local_clouds, to_remove)
        else:
            clean_dates = image_dates
            clean_filenames = filenames
        if len(clean_dates) >= 11:
            clean_dates = np.delete(clean_dates, 5)
            clean_filenames = np.delete(clean_filenames, 5)
            cloud_probs = np.delete(cloud_probs, 5, 0)
            cloud_percent = np.delete(cloud_percent, 5)
            local_clouds = np.delete(local_clouds, to_remove)

        _ = cloud_removal.print_dates(
            clean_dates, cloud_percent
        )

        # Expand the image selection if multiple of the local cloud
        # prob is above a threshold
        lowest_three_local = np.argpartition(all_local_clouds, 3)[:3]
        lowest_four_local = np.argpartition(all_local_clouds, 4)[:4]
        #lowest_ten_local = np.argpartition(all_local_clouds, 10)[:10]

        # If less than 3 images selected have <40% local cloud cover,
        # Then expand the image search
        criteria1 = (np.sum((local_clouds <= 0.3)) < 3)
        criteria2 = (np.sum((local_clouds <= 0.4)) < 4)
        criteria3 = len(local_clouds) <= 8
        criteria2 = np.logical_or(criteria2, criteria3)
        if criteria1 or criteria2:
            if len(clean_dates) <= 9:
                lowest = lowest_four_local if criteria2 else lowest_three_local
                lowest_dates = image_dates[lowest]
                lowest_filenames = filenames[lowest]
                existing_imgs_in_local = [x for x in clean_dates if x in image_dates[lowest]]
                images_to_add = [x for x in lowest_dates if x not in clean_dates]
                filenames_to_add = [x for x in lowest_filenames if x not in clean_filenames]
            
                clean_dates = np.concatenate([clean_dates, images_to_add])
                clean_filenames = np.concatenate([clean_filenames, filenames_to_add])
                clean_dates = np.sort(clean_dates)
            if len(clean_dates) <= 9:
                imgs_to_add = 9 - len(clean_dates)
                
                #print(f"[DEBUG] all_local_clouds length: {len(all_local_clouds)}")
                #print(f"[DEBUG] all_local_clouds: {all_local_clouds}")

                if len(all_local_clouds) > 5:
                    lowest_five_local = np.argpartition(all_local_clouds, 5)[:5]
                    #print(f"[DEBUG] Using argpartition. lowest_five_local indices: {lowest_five_local}")
                else:
                    lowest_five_local = np.argsort(all_local_clouds)[:len(all_local_clouds)]
                    #print(f"[DEBUG] Using argsort fallback. lowest_five_local indices: {lowest_five_local}")
                
                images_to_add = [x for x in image_dates[lowest_five_local] if x not in clean_dates][:imgs_to_add]
                filenames_to_add = [x for x in filenames[lowest_five_local] if x not in clean_filenames][:imgs_to_add]

                #print(f"[DEBUG] images_to_add: {images_to_add}")
                #print(f"[DEBUG] filenames_to_add: {filenames_to_add}")

                clean_dates = np.concatenate([clean_dates, images_to_add])
                clean_filenames = np.concatenate([clean_filenames, filenames_to_add])
                clean_dates = np.sort(clean_dates)
        #for i, x, y in zip(clean_dates, cloud_percent, local_clouds):
        #    print(i, x, y)

        hkl.dump(cloud_probs, clouds_file, mode='w', compression='gzip')
        hkl.dump(clean_dates, clean_steps_file, mode='w', compression='gzip')
        hkl.dump(clean_filenames, clean_filename, mode='w', compression='gzip')
    else:
        clean_dates =  np.arange(0, 9)
 
    if not (os.path.exists(s2_10_file)):

        clean_steps = hkl.load(clean_steps_file)
        cloud_probs = hkl.load(clouds_file)
        clean_filenames = hkl.load(clean_filename)
        
        
        s2_10, s2_20, clm,s2_dates = gee_downloading.download_sentinel_2_new(clean_filenames,initial_bbx,
                                                     dates = dates,
                                                     year = year,crs = crs, maxclouds = 0.3)
 

        # Ensure that L2A, L1C derived products have exact matching dates
        # As sometimes the L1C data has more dates than L2A if processing bug from provider
        to_remove_clouds = [i for i, val in enumerate(clean_steps) if val not in s2_dates]
        to_remove_dates = [val for i, val in enumerate(clean_steps) if val not in s2_dates]

        # Save all the files to disk (temporarily)
        hkl.dump(to_int16(s2_10), s2_10_file, mode='w', compression='gzip')
        hkl.dump(to_int16(s2_20), s2_20_file, mode='w', compression='gzip')
        hkl.dump(np.array(clean_steps), s2_dates_file, mode='w', compression='gzip')
        hkl.dump(clm, cloud_mask_file, mode='w', compression='gzip')
        # We need to know the size to ensure that Sentinel-1 is the same size as
        # Sentinel-2
        size = s2_20.shape[1:3]

            
    if not (os.path.exists(s1_file)) and len(clean_dates) > 2:
        print("Downloading sentinel 1 files")
        
        download_s1_tile_gee(initial_bbx,year,s1_file,s1_dates_file,crs)
        

    if not os.path.exists(dem_file) and len(clean_dates) > 2:
        print(f'Downloading DEM file')
        
        dem = gee_downloading.download_dem(initial_bbx,crs = crs)
        
        hkl.dump(dem, dem_file, mode='w', compression='gzip')


    return myBox, len(clean_dates), crs


def upload_to_gcs(local_file_path, bucket_name, blob_path, year, project_id=""):
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)

    # build a year‐folder under your blob_path
    target_path = f"{year}/{blob_path}"

    blob = bucket.blob(target_path)
    blob.upload_from_filename(local_file_path)

    print(f"✅ Uploaded {local_file_path} to gs://{bucket_name}/{target_path}")

def write_geotiff(bbox, crs, height, width, out_path,data):
    """
    Save a dummy GeoTIFF with the specified bounding box, CRS, and shape.
    
    Parameters:
        bbox (list): [min_lon, min_lat, max_lon, max_lat]
        crs (str or dict): CRS string (e.g., "EPSG:4326") or rasterio CRS object
        height (int): number of rows
        width (int): number of columns
        out_path (str): output file path
    """
    # Create dummy data
    dummy_data = data 

    # Create affine transform
    transform = from_bounds(*bbox, width, height)

    # Save dummy GeoTIFF
    with rasterio.open(
        out_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=dummy_data.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(dummy_data, 1)

    print(f"Dummy GeoTIFF written to {out_path}")



#cfg = load_config("config.yaml")
#print(cfg)
local_path = f"/tmp/app/" #f"{cfg['paths']['local_path']}/"
superresolve_model_path = "/app/supres-40k-swir/" #cfg["paths"]["superresolve_model_path"]



#parser = argparse.ArgumentParser(description="Append a labeled point to the labeltcc file.")
#parser.add_argument("--lat", type=float, required=True, help="Latitude value")
#parser.add_argument("--lon", type=float, required=True, help="Longitude value")
#parser.add_argument("--x", type=int, default=1, help="X coordinate (default: 1)")
#parser.add_argument("--y", type=int, default=1, help="Y coordinate (default: 1)")
#parser.add_argument("--country", type=str, default="cameroon", help="country name")
#parser.add_argument("--year", type=int, default=2024, help="year")
#parser.add_argument("--outputdir", type=str, default="/home/ate/temp/", help="output dir")
#parser.add_argument('--ee-key', type=str, help='Path to Earth Engine service account key')

#args = parser.parse_args()

#import ee
#import google.auth
#from google.oauth2 import service_account

#credentials = service_account.Credentials.from_service_account_file(
#     cfg["paths"]['ee-key'],
#    scopes=["https://www.googleapis.com/auth/earthengine.readonly"]
#)
#ee.Initialize(credentials)



# Lots of code here to load two tensorflow graphs at once
superresolve_graph_def = tf.compat.v1.GraphDef()
predict_graph_def = tf.compat.v1.GraphDef()
if os.path.exists(superresolve_model_path):
    print(f"Loading model from {superresolve_model_path}")
    superresolve_file = tf.io.gfile.GFile(superresolve_model_path + "superresolve_graph.pb", 'rb')
    superresolve_graph_def.ParseFromString(superresolve_file.read())
    superresolve_graph = tf.import_graph_def(superresolve_graph_def, name='superresolve')
    superresolve_sess = tf.compat.v1.Session(graph=superresolve_graph)
    superresolve_logits = superresolve_sess.graph.get_tensor_by_name("superresolve/Add_2:0")
    superresolve_inp = superresolve_sess.graph.get_tensor_by_name("superresolve/Placeholder:0")
    superresolve_inp_bilinear = superresolve_sess.graph.get_tensor_by_name("superresolve/Placeholder_1:0")
else:
    raise Exception(f"The model path {superresolve_model_path} does not exist")

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)



def process_tile_row(lat, lon, x, y, country, province, year, bucket_name, blob_path,project):

    try:
        client = storage.Client(project)
        bucket = client.bucket(bucket_name)

        # build the GCS keys
        geotiff_blob = f"{year}/{blob_path}/{y}_{x}.tif"
        npz_blob     = f"{year}/{blob_path}/{y}_{x}.npz"

        if bucket.blob(npz_blob).exists():
            print(f"↩️  Skipping {x},{y}: already in gs://{bucket_name}/{year}/{blob_path}")
            return

        initial_bbx = [lon, lat, lon, lat]
        expansion = 200
        print("############# step 1: getting the data #############")
        bbx, n_images, crs = download_tile(x = x, y = y,  year = year, initial_bbx = initial_bbx, expansion = expansion,local_path =  local_path )

        out_tif_path = f"{local_path}/{str(year)}/geotifs/{y}_{x}.tif"
        write_geotiff(bbx, crs, 640, 640, out_tif_path,np.zeros((640, 640), dtype=np.uint8))
        upload_to_gcs(out_tif_path, bucket_name, f"{blob_path}/{y}_{x}.tif",year)

        #print("############# step 1a: getting the label #############")

        #label = gee_downloading.getLabel(initial_bbx,crs)
        #out_tif_path = f"{outputdir}/{str(year)}/label/{y}_{x}.tif"
        #print(label)
        #write_geotiff(bbx, crs, 640, 640, out_tif_path,label)

        print("############# step 2: processing data #############")
        s2, dates, interp, s1, dem, cloudshad, snow = process_tile(x = x,y = y,local_path = local_path, bbx = bbx, make_shadow = True)

        print("############# step 3: apply super resolve #############")
        s2[..., :10] = superresolve_large_tile(s2[..., :10], superresolve_sess)


        print("############# step 4 process and store tiles #############")
        full_tile = process_subtiles(local_path,x, y, s2, dates, interp, s1, dem, bbx, SIZE, bbx)

        print("############# step 5 store training data #############")
        #full_tile = process_subtiles_label(local_path,x, y, s2, dates, interp, s1, dem, bbx, SIZE, bbx)


        print("############# step 6 store data #############")
        scaled = np.round(full_tile * 10000).astype(np.int16)

        outfile = f"{local_path}/{str(year)}/data/{y}_{str(x)}.npz"
        print(f"saving file {outfile}")

        np.savez_compressed(outfile, data=scaled)
        upload_to_gcs(outfile, bucket_name, f"{blob_path}/{y}_{str(x)}.npz",year)
    except:
        pass
        
#cfg = load_config()

parser = argparse.ArgumentParser()

#parser.add_argument("--x", type=int, required=True)
#parser.add_argument("--y", type=int, required=True)
#parser.add_argument("--lon", type=float, required=True)
#parser.add_argument("--lat", type=float, required=True)
parser.add_argument("--year", type=int, required=True)
#parser.add_argument("--outputdir", type=str, default=cfg["paths"]["output_dir"])
parser.add_argument("--bucket_name", type=str, required=True)
parser.add_argument("--tile_list", type=str, required=True)
parser.add_argument("--blob_path", type=str, default="cameroon")
#parser.add_argument("--country", type=str, default="cameroon")
parser.add_argument("--project", type=str, default="")


args = parser.parse_args()

year = args.year
#lat = args.lat
#lon = args.lon
#x = args.x
#y = args.y
#country = args.country
outputdir = "/tmp/app/"
bucket_name = args.bucket_name
blob_path = args.blob_path
project = args.project





# Create directories if they don't exist
os.makedirs(f"{local_path}/{str(year)}/geotifs", exist_ok=True)
os.makedirs(f"{local_path}/{str(year)}/label", exist_ok=True)
os.makedirs(f"{local_path}/{str(year)}/data", exist_ok=True)


if args.tile_list:
    # read the CSV (local or gs:// path—pandas can handle both with gcsfs installed)
    df = pd.read_csv(args.tile_list)
    print(df)
    for _, row in df.iterrows():
        print(row)
        process_tile_row(
            lat=row["lat"],
            lon=row["lon"],
            x=int(row["x"]),
            y=int(row["y"]),
            country=row.get("country", ""),      # only if you need it
            province=row.get("province", ""),
            year=year,
            bucket_name=bucket_name,
            blob_path=blob_path,
            project = project
        )


'''
print("############# step 1: getting the data #############")
bbx, n_images, crs = download_tile(x = x, y = y,  year = year, initial_bbx = initial_bbx, expansion = expansion,local_path =  local_path )

out_tif_path = f"{local_path}/{str(year)}/geotifs/{y}_{x}.tif"
write_geotiff(bbx, crs, 640, 640, out_tif_path,np.zeros((640, 640), dtype=np.uint8))
upload_to_gcs(out_tif_path, bucket_name, f"{blob_path}/{y}_{x}.tif",year)

#print("############# step 1a: getting the label #############")

#label = gee_downloading.getLabel(initial_bbx,crs)
#out_tif_path = f"{outputdir}/{str(year)}/label/{y}_{x}.tif"
#print(label)
#write_geotiff(bbx, crs, 640, 640, out_tif_path,label)

print("############# step 2: processing data #############")
s2, dates, interp, s1, dem, cloudshad, snow = process_tile(x = x,y = y,local_path = local_path, bbx = bbx, make_shadow = True)

print("############# step 3: apply super resolve #############")
s2[..., :10] = superresolve_large_tile(s2[..., :10], superresolve_sess)


print("############# step 4 process and store tiles #############")
full_tile = process_subtiles(local_path,x, y, s2, dates, interp, s1, dem, bbx, SIZE, bbx)

print("############# step 5 store training data #############")
#full_tile = process_subtiles_label(local_path,x, y, s2, dates, interp, s1, dem, bbx, SIZE, bbx)


print("############# step 6 store data #############")
scaled = np.round(full_tile * 10000).astype(np.int16)

outfile = f"{local_path}/{str(year)}/data/{y}_{str(x)}.npz"
print(f"saving file {outfile}")

np.savez_compressed(outfile, data=scaled)
upload_to_gcs(outfile, bucket_name, f"{blob_path}/{y}_{str(x)}.npz",year)
'''
