import os
import numpy as np
import rasterio


def write_tif(arr: np.ndarray,
              point: list,
              x: int,
              y: int,
              out_folder: str,
              suffix="_FINAL") -> str:
    #! TODO: Documentation

    file = out_folder + f"{str(x)}X{str(y)}Y{suffix}.tif"

    west, east = point[0], point[2]
    north, south = point[3], point[1]
    arr = arr.T.astype(np.uint8)

    transform = rasterio.transform.from_bounds(west=west,
                                               south=south,
                                               east=east,
                                               north=north,
                                               width=arr.shape[1],
                                               height=arr.shape[0])

    print("Writing", file)
    new_dataset = rasterio.open(file,
                                'w',
                                driver='GTiff',
                                height=arr.shape[0],
                                width=arr.shape[1],
                                count=1,
                                dtype="uint8",
                                compress='lzw',
                                crs='+proj=longlat +datum=WGS84 +no_defs',
                                transform=transform)
    new_dataset.write(arr, 1)
    new_dataset.close()
    return file

def make_output_and_temp_folders(output_folder: str) -> None:
    """Makes necessary folder structures for input/output of raw/processed data

        Parameters:
         idx (str)
         output_folder (path)

        Returns:
         None
    """

    def _find_and_make_dirs(dirs: list) -> None:
        if not os.path.exists(os.path.realpath(dirs)):
            os.makedirs(os.path.realpath(dirs))

    folders = [
        'raw/', 'raw/clouds/', 'raw/misc/', 'raw/s1/', 'raw/s2_10/',
        'raw/s2_20/'
    ]

    for folder in folders:
        _find_and_make_dirs(output_folder + folder)

def make_subtiles(folder: str, tiles) -> None:

    y_tiles = np.unique(tiles[:, 1])
    x_tiles = np.unique(tiles[:, 0])

    def _find_and_make_dirs(dirs):
        if not os.path.exists(os.path.realpath(dirs)):
            os.makedirs(os.path.realpath(dirs))

    for y_tile in y_tiles:
        _find_and_make_dirs(folder + str(y_tile) + '/')
