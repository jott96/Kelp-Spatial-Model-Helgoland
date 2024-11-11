import warnings
warnings.filterwarnings("ignore")

##################################################################################################################
# Import libraries and file paths ################################################################################
##################################################################################################################

'''
This section ensures that all necessary python libraries are loaded and that the file management is set up correctly.
'''

# load python packages
import geopandas as gpd
import pandas as pd
import rasterio as rio
from rasterio import Affine
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import rasterio.features
from shapely.geometry import shape
import seaborn as sns
from rasterio.mask import mask
from rasterio.plot import show
from shapely.geometry import box
import tempfile
import colorsys
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
import optuna
import shap
from itertools import combinations
from sklearn.linear_model import LinearRegression
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy.ndimage import median_filter
from scipy.stats import spearmanr


# set path of kelp field data table
kelp_path = r"M:\KELP_Helgoland\data\final_input\24-04-11_kelp_master_table_reworked.gpkg"

#### set paths for all predictor datasets #### 
# water depth
bathymetry_path = r"M:\KELP_MA\data\auxiliary_data\bathymetry\24-07-24_Bathy_BAW2020_MLLW_5m_aoi.tif"

# substrate and terrain
subs_path = r"M:\KELP_MA\data\auxiliary_data\substrate\24-13-05_pred_subs_OK_5x5m_all_considered30.tif"
aspect_path = r"M:\KELP_MA\data\auxiliary_data\terrain\24-06-26_aspect_from_BAW_bathy_5m.tif"
slope_path = r"M:\KELP_MA\data\auxiliary_data\terrain\24-06-26_slope_from_BAW_bathy_5m.tif"
tpi_path = r"M:\KELP_MA\data\auxiliary_data\terrain\24-06-26_TPI_from_BAW_bathy_5m.tif"

# S2 time series - 2023
S2_b2_med_path = r"M:\KELP_MA\data\auxiliary_data\RS_data\S2\GEE_time_series\median\2023\S2_b2_0401-3108_med.tif"
S2_b3_med_path = r"M:\KELP_MA\data\auxiliary_data\RS_data\S2\GEE_time_series\median\2023\S2_b3_0401-3108_med.tif"
S2_b4_med_path = r"M:\KELP_MA\data\auxiliary_data\RS_data\S2\GEE_time_series\median\2023\S2_b4_0401-3108_med.tif"
S2_b8_med_path = r"M:\KELP_MA\data\auxiliary_data\RS_data\S2\GEE_time_series\median\2023\S2_b8_0401-3108_med.tif"
S2_ndvi_med_path = r"M:\KELP_MA\data\auxiliary_data\RS_data\S2\GEE_time_series\median\2023\S2_ndvi_0401-3108_med.tif"
S2_ndavi_med_path = r"M:\KELP_MA\data\auxiliary_data\RS_data\S2\GEE_time_series\median\2023\S2_ndavi_0401-3108_med.tif"
S2_wavi_med_path = r"M:\KELP_MA\data\auxiliary_data\RS_data\S2\GEE_time_series\median\2023\S2_wavi_0401-3108_med.tif"
S2_grvi_med_path = r"M:\KELP_MA\data\auxiliary_data\RS_data\S2\GEE_time_series\median\2023\S2_grvi_0401-3108_med.tif"
S2_npci_med_path = r"M:\KELP_MA\data\auxiliary_data\RS_data\S2\GEE_time_series\median\2023\S2_npci_0401-3108_med.tif"
S2_bg_med_path = r"M:\KELP_MA\data\auxiliary_data\RS_data\S2\GEE_time_series\median\2023\S2_bg_ratio_0401-3108_med.tif"

# wave data
wv_path = r"M:\KELP_MA\data\auxiliary_data\waves\24-07-23_bwv_KNNimputed_5mResampled_240401-24080_daily_med.tif"

# set path for PS scene - 20230708
PS_path = r"M:\KELP_MA\data\auxiliary_data\RS_data\PlanetScope\KELP_2023_psscene_analytic_8b_sr_udm2\PSScene\20230708_103224_63_2402_3B_AnalyticMS_SR_8b_clip.tif"

# set paths for S2 scene - 20230909
S2_band_paths = [
    r"M:\KELP_MA\data\auxiliary_data\RS_data\S2\S2A_MSIL2A_20230909T103631_N0509_R008_T32UMF_20230909T155201.SAFE\GRANULE\L2A_T32UMF_A042905_20230909T104403\IMG_DATA\R10m\T32UMF_20230909T103631_B02_10m.jp2",
    r"M:\KELP_MA\data\auxiliary_data\RS_data\S2\S2A_MSIL2A_20230909T103631_N0509_R008_T32UMF_20230909T155201.SAFE\GRANULE\L2A_T32UMF_A042905_20230909T104403\IMG_DATA\R10m\T32UMF_20230909T103631_B03_10m.jp2",
    r"M:\KELP_MA\data\auxiliary_data\RS_data\S2\S2A_MSIL2A_20230909T103631_N0509_R008_T32UMF_20230909T155201.SAFE\GRANULE\L2A_T32UMF_A042905_20230909T104403\IMG_DATA\R10m\T32UMF_20230909T103631_B04_10m.jp2",
    r"M:\KELP_MA\data\auxiliary_data\RS_data\S2\S2A_MSIL2A_20230909T103631_N0509_R008_T32UMF_20230909T155201.SAFE\GRANULE\L2A_T32UMF_A042905_20230909T104403\IMG_DATA\R10m\T32UMF_20230909T103631_B08_10m.jp2"
]

# set paths for specific geometries (used for map generation)
helgoland_polygon_path = r"M:\KELP_Helgoland\data\helgoland_polygon\helgoland_polygon_32632.gpkg"
isolines_path = r"M:\KELP_MA\data\specific_geometries\24-07-24_1m_isolines_bathymetryBAW2020_MLLW_aoi.gpkg"

# set current date and output mode for plotting
date = '24-11-11'
output_mode = False

#### prepare further geometries for fancy-looking plots ####
# drop the 'Wellenbrecher' polygon
helgo_poly = gpd.read_file(helgoland_polygon_path)
helgo_poly = helgo_poly[helgo_poly["Gebiet"] != "Wellenbrecher_Duene"]
# load in 5 m isolines and filter for the relevant ones
isolines = gpd.read_file(isolines_path)
isolines_crs = isolines.crs
isolines_filtered = isolines[isolines['ELEV'].isin([-5, -10, -15])]

#%%
##################################################################################################################
# Load bathymetry and kelp gdf as basesets for the whole approach ####### ########################################
##################################################################################################################

# load in the kelp data as gdf (field data from kelp monitoring 2023)
kelp_gdf = gpd.read_file(kelp_path)

# open the bathymetry raster to get its data, extent and crs; this defines the AOI
with rio.open(bathymetry_path) as bathymetry_src:
    bathymetry_extent = bathymetry_src.bounds
    bathymetry_data = bathymetry_src.read(1)
    nodata_value = bathymetry_src.nodata
    bathymetry_crs = bathymetry_src.crs
    print("Bathymetry CRS:", bathymetry_crs)
    target_profile = bathymetry_src.profile # define the profile for the output raster (from bathy raster)
    target_profile.update(count=1, dtype=rasterio.float32, nodata=np.nan)

##################################################################################################################
# Load S2 terrain and substrate data #############################################################################
##################################################################################################################

# open the krigged substrate solidity raster (OK was performed in R) 
with rio.open(subs_path) as subs_src:
    subs_data = subs_src.read(1)
# open the derived slope raster (generated in QGIS) 
with rio.open(slope_path) as slope_src:
    slope_data = slope_src.read(1)
# open the derived aspect raster (generated in QGIS) --> transform angle into range from -1 (north) to +1 (south) using negative cosine
with rio.open(aspect_path) as aspect_src:
    aspect_data = aspect_src.read(1)
    aspect_radians = np.deg2rad(aspect_data) # convert data to radians
    aspect_data = -np.cos(aspect_radians)
# open the derived TPI raster (generated in QGIS) 
with rio.open(tpi_path) as tpi_src:
    tpi_data = tpi_src.read(1)
    
##################################################################################################################
# Load wave data #################################################################################################
##################################################################################################################

# open the generated bottom wave velocity raster (generated in custom script) 
with rio.open(wv_path) as wv_src:
    wv_data = wv_src.read(1)

##################################################################################################################
# Load S2 time series data #######################################################################################
##################################################################################################################

'''
In this section, the S2 time series data gets loaded. The median values for the different bands and indicices originate
from a time series between spring and summer 2023 which comprises fourteen S2 images.
'''

### load S2 time series data derived from GEE ###  
# open the S2 blue band median
with rio.open(S2_b2_med_path) as S2_blue_src:
    S2_blue_med_data = S2_blue_src.read(1)
# open the S2 green band median
with rio.open(S2_b3_med_path) as S2_green_src:
    S2_green_med_data = S2_green_src.read(1)
# open the S2 red band median
with rio.open(S2_b4_med_path) as S2_red_src:
    S2_red_med_data = S2_red_src.read(1)
# open the S2 nir band median
with rio.open(S2_b8_med_path) as S2_nir_src:
    S2_nir_med_data = S2_nir_src.read(1)
# open the S2 ndvi median raster
with rio.open(S2_ndvi_med_path) as S2_ndvi_src:
    S2_ndvi_med_data = S2_ndvi_src.read(1)
# open the S2 ndavi median raster 
with rio.open(S2_ndavi_med_path) as ndavi_src:
    S2_ndavi_med_data = ndavi_src.read(1)
# open the S2 wavi median raster 
with rio.open(S2_wavi_med_path) as wavi_src:
    S2_wavi_med_data = wavi_src.read(1)
# open the S2 grvi median raster 
with rio.open(S2_grvi_med_path) as grvi_src:
    S2_grvi_med_data = grvi_src.read(1) 
# open the S2 npci median raster 
with rio.open(S2_npci_med_path) as npci_src:
    S2_npci_med_data = npci_src.read(1)
# open the S2 bg ratio median raster
with rio.open(S2_bg_med_path) as bg_src:
    S2_bg_med_data = bg_src.read(1)

##################################################################################################################
# Load and process RS single scene imagery (S2/PS) ###############################################################
##################################################################################################################

'''
In this section, the spectral bands of the S2 and PS monotemporal imagery are loaded. They are cropped to the
spatial extent of the AOI and are resampled to a spatial resolution of 5 x 5 m, which is set
the target resolution of the final model. For each image, a band stack is created. 
'''

### Sentinel-2 ###
# resample all S2 bands to 5 m resolution and clip them to the bathymetry raster's extent (AOI)
# set target resolution in meters
target_resolution = (5, 5)  
# initialize an empty list that will store the resampled and clipped bands
S2_clipped_bands = []

# loop over S-2 bands blue, green, red and nir
for S2_band_path in S2_band_paths:
    with rio.open(S2_band_path) as src:
        # resample the band to the target resolution
        resampled_band, resampled_transform = rio.warp.reproject(
            source=src.read(),  # read the entire band
            src_transform=src.transform,
            src_crs=src.crs,
            dst_crs=src.crs,  # set destination CRS same as source CRS
            dst_resolution=target_resolution, # resampling from 10 m to 5 m
            resampling=rio.enums.Resampling.bilinear 
        )
        # create a rasterio dataset from the resampled band - was needed to avoid 'nodata-error' in clipping 
        resampled_band_dataset = rasterio.open(
            'temp.tif',
            'w',
            driver='GTiff',
            height=resampled_band.shape[1],
            width=resampled_band.shape[2],
            count=1,
            dtype=resampled_band.dtype,
            crs=src.crs,
            transform=resampled_transform
        )
        resampled_band_dataset.write(resampled_band)
        resampled_band_dataset.close()
        # open the resampled band dataset in read mode
        with rasterio.open('temp.tif') as resampled_band_dataset:
            # clip the resampled band to the extent of the bathymetry raster (AOI)
            clipped_band, clipped_transform = mask(resampled_band_dataset, [box(*bathymetry_extent)], crop=True)     
        # update metadata for the clipped band
        clipped_meta = {
            "transform": clipped_transform,
            "height": clipped_band.shape[1],
            "width": clipped_band.shape[2],
            "nodata": None  # update nodata value if necessary
        }      
        # append the clipped band to the list of clipped bands and rescale them to %-reflectance
        S2_clipped_bands.append(clipped_band / 10000)

# stack the clipped bands for S2
S2_stacked_bands = np.stack(S2_clipped_bands) 
# remove singleton dimension
S2_stacked_bands = np.squeeze(S2_stacked_bands)

# # apply mean filtering to the bands (disabled for final thesis modeling)
# mean_kernel = np.ones((3, 3)) / 9 # 3x3 mean kernel
# S2_filtered_stacked_bands = []
# for S2_band in S2_stacked_bands:
#     S2_filtered_band = scipy.ndimage.convolve(S2_band, mean_kernel)
#     S2_filtered_stacked_bands.append(S2_filtered_band)
# S2_filtered_stacked_bands = np.stack(S2_filtered_stacked_bands)
# S2_stacked_bands = S2_filtered_stacked_bands # rename to 'stacked_bands' to simplify for further processing


### PlanetScope ###
# reproject and resample all PS bands to 5 m resolution; clip them to the bathymetry rater (AOI) 
# set target resolution in meters
target_resolution = (5, 5)
with rio.open(PS_path) as src:
    # get the CRS of the PlanetScope data
    PS_crs = src.crs
    # reproject each band to the CRS of the bathymetry raster
    # initialize an empty list that will store the processed bands
    clipped_bands = []
    for i in [2, 4, 6, 8]:  # PlanetScope bands: 2 (blue), 4 (green), 6 (red), 8 (NIR) - comparable to S2
        resampled_band, resampled_transform = rasterio.warp.reproject(
            source=src.read(i),  # read the specific band
            src_transform=src.transform,
            src_crs=src.crs,
            dst_crs=bathymetry_crs,  # set destination CRS to the bathymetry CRS
            dst_resolution=target_resolution, # resampling from 3 m to 5 m
            resampling=rio.enums.Resampling.bilinear
        )
        # create temporary rasterio dataset from the resampled band - again necessary to avoid 'nodata-error' in clipping
        resampled_band_dataset = rasterio.open(
            'temp.tif',
            'w',
            driver='GTiff',
            height=resampled_band.shape[1],
            width=resampled_band.shape[2],
            count=1,
            dtype=resampled_band.dtype,
            crs=src.crs,
            transform=resampled_transform
        )
        resampled_band_dataset.write(resampled_band)
        resampled_band_dataset.close()
        # open the resampled band dataset in read mode
        with rasterio.open('temp.tif') as resampled_band_dataset:
            # clip the resampled band to the extent of the bathymetry raster (AOI)
            clipped_band, clipped_transform = mask(resampled_band_dataset, [box(*bathymetry_extent)], crop=True)     
        # update metadata for the clipped band
        clipped_meta = {
            "transform": clipped_transform,
            "height": clipped_band.shape[1],
            "width": clipped_band.shape[2],
            "nodata": None  # update nodata value if necessary
        }      
        # slice out the empty col and row to assure congruency and append the clipped band to the list of clipped bands
        clipped_band = np.delete(clipped_band, -1, axis=1)  # remove the lowermost row
        clipped_band = np.delete(clipped_band, 0, axis=2)  # remove the leftmost column 
        clipped_bands.append(clipped_band / 1000)


# stack the clipped bands for PS
PS_stacked_bands = np.stack(clipped_bands) 
# remove singleton dimension
PS_stacked_bands = np.squeeze(PS_stacked_bands)

# # apply mean filtering to the bands
# # 3x3 doesn't seems to make a big diff on S2 scene
# mean_kernel = np.ones((3, 3)) / 9 # 3x3 mean kernel
# filtered_stacked_bands = []
# for band in stacked_bands:
#     filtered_band = scipy.ndimage.convolve(band, mean_kernel)
#     filtered_stacked_bands.append(filtered_band)
# filtered_stacked_bands = np.stack(filtered_stacked_bands)
# stacked_bands = filtered_stacked_bands # rename to 'stacked_bands' to simplify for further processing

# set the CRS for the clipped metadata
clipped_meta['crs'] = kelp_gdf.crs

# calculate extent for plotting from the stored transform
xmin = clipped_meta['transform'][2]
xmax = clipped_meta['transform'][2] + clipped_meta['transform'][0] * clipped_meta['width']
ymin = clipped_meta['transform'][5] + clipped_meta['transform'][4] * clipped_meta['height']
ymax = clipped_meta['transform'][5]

# plot RGB composite
rgb = PS_stacked_bands[[2, 1, 0], :, :]  # reorder bands to RGB
rgb = np.moveaxis(rgb, 0, -1)  # move bands axis to last dimension
plt.figure(figsize=(10, 8))
plt.imshow(rgb * 1, extent=[xmin, xmax, ymin, ymax])
plt.title('S2 RGB Composite')
plt.xlabel('Northing')
plt.ylabel('Easting')
# plot the GeoDataFrame as dot markers on top of the rgb composite
plt.scatter(kelp_gdf.geometry.x, kelp_gdf.geometry.y, color='red', s=5, label='Mapping sites')
plt.legend()
plt.show()

#%%
##################################################################################################################
# Generate indices and the final data stack ######################################################################
##################################################################################################################

'''
In this section, spectral indices are derived from the monotemporal S2 and PS imagery and all loaded and 
processed predictors are combined into a data stack which enables further selection/deselection of predictors
during model evaluation loops.
'''

# define functions to compute indices
# NDVI - Normalized Difference Vegetation Index
def compute_ndvi(nir, red):
    return (nir - red) / (nir + red)
# NDAVI - Normalized Difference Aquatic Vegetation Index (Villa et al. 2014)
def compute_ndavi(blue, nir):
    return (nir - blue) / (nir + blue)  
# Water Adjusted Vegetation Index
def compute_wavi(blue, nir , LW=0.5): # LW is background correction factor (Villa et al. 2014)
    return (1+LW) * (nir - blue) / (nir + blue + LW)  
# GRVI - Green-Red Vegetation Index
def compute_grvi(green, red):
    return (green - red) / (green + red)  
# Normalized Total Pigmet to Chl-a Ratio Index
def compute_npci(blue, red):
    return (red - blue) / (red + blue)  
# Simple Ratio - BlueGreen
def compute_diff_BG(blue, green):
    return blue / green

### Sentinel-2 ###
# compute indices for each pixel based on the single RS scene
S2_ndvi = compute_ndvi(S2_stacked_bands[3], S2_stacked_bands[2])  # nir and red band
S2_ndavi = compute_ndavi(S2_stacked_bands[0], S2_stacked_bands[3]) # blue and nir band
S2_wavi = compute_wavi(S2_stacked_bands[0], S2_stacked_bands[3]) # blue and nir band
S2_grvi = compute_grvi(S2_stacked_bands[1], S2_stacked_bands[2]) # green and red band
S2_npci = compute_npci(S2_stacked_bands[0], S2_stacked_bands[2]) # blue and red band
S2_bg = compute_diff_BG(S2_stacked_bands[0], S2_stacked_bands[1]) # blue and green band

### PlanetScope ###
# compute indices for each pixel based on the single RS scene
PS_ndvi = compute_ndvi(PS_stacked_bands[3], PS_stacked_bands[2])  # nir and red band
PS_ndavi = compute_ndavi(PS_stacked_bands[0], PS_stacked_bands[3]) # blue and nir band
PS_wavi = compute_wavi(PS_stacked_bands[0], PS_stacked_bands[3]) # blue and nir band
PS_grvi = compute_grvi(PS_stacked_bands[1], PS_stacked_bands[2]) # green and red band
PS_npci = compute_npci(PS_stacked_bands[0], PS_stacked_bands[2]) # blue and red band
PS_bg = compute_diff_BG(PS_stacked_bands[0], PS_stacked_bands[1]) # blue and green band

### reshape the arrays to make them stackable with the bands stack ### 
bathymetry_data = bathymetry_data[np.newaxis, ...] # add new axis for bands
subs_data = subs_data[np.newaxis, ...] # add new axis for bands
# S-2 single scene
S2_ndvi_reshaped = S2_ndvi[np.newaxis, ...]
S2_ndavi_reshaped = S2_ndavi[np.newaxis, ...]
S2_wavi_reshaped = S2_wavi[np.newaxis, ...]
S2_grvi_reshaped = S2_grvi[np.newaxis, ...]
S2_npci_reshaped = S2_npci[np.newaxis, ...]
S2_bg_reshaped = S2_bg[np.newaxis, ...]
# PS single scene
PS_ndvi_reshaped = PS_ndvi[np.newaxis, ...]
PS_ndavi_reshaped = PS_ndavi[np.newaxis, ...]
PS_wavi_reshaped = PS_wavi[np.newaxis, ...]
PS_grvi_reshaped = PS_grvi[np.newaxis, ...]
PS_npci_reshaped = PS_npci[np.newaxis, ...]
PS_bg_reshaped = PS_bg[np.newaxis, ...]
# S-2 time series (April - August)
S2_blue_med_reshaped = S2_blue_med_data[np.newaxis, ...]
S2_green_med_reshaped = S2_green_med_data[np.newaxis, ...]
S2_red_med_reshaped = S2_red_med_data[np.newaxis, ...]
S2_nir_med_reshaped = S2_nir_med_data[np.newaxis, ...]
S2_ndvi_med_reshaped = S2_ndvi_med_data[np.newaxis, ...]
S2_ndavi_med_reshaped = S2_ndavi_med_data[np.newaxis, ...]
S2_wavi_med_reshaped = S2_wavi_med_data[np.newaxis, ...]
S2_grvi_med_reshaped = S2_grvi_med_data[np.newaxis, ...]
S2_npci_med_reshaped = S2_npci_med_data[np.newaxis, ...]
S2_bg_med_reshaped = S2_bg_med_data[np.newaxis, ...]
# terrain 
slope_reshaped = slope_data[np.newaxis, ...]
aspect_reshaped = aspect_data[np.newaxis, ...]
tpi_reshaped = tpi_data[np.newaxis, ...]
# waves
wv_reshaped = wv_data[np.newaxis, ...]


# add computed indices to the stack of clipped bands --> this stack is the base for the following spatial model
data_stack = np.vstack([
                        bathymetry_data * -1, # inverted to have positive numbers for water depth
                        subs_data,
                        slope_reshaped,
                        aspect_reshaped,
                        tpi_reshaped,
                        wv_reshaped,
                        S2_stacked_bands,
                        S2_ndvi_reshaped, 
                        S2_ndavi_reshaped,
                        S2_wavi_reshaped,
                        S2_grvi_reshaped,
                        S2_npci_reshaped,
                        S2_bg_reshaped,
                        PS_stacked_bands,
                        PS_ndvi_reshaped, 
                        PS_ndavi_reshaped,
                        PS_wavi_reshaped,
                        PS_grvi_reshaped,
                        PS_npci_reshaped,
                        PS_bg_reshaped,
                        S2_blue_med_reshaped,
                        S2_green_med_reshaped,
                        S2_red_med_reshaped,
                        S2_nir_med_reshaped,
                        S2_ndvi_med_reshaped, 
                        S2_ndavi_med_reshaped,
                        S2_wavi_med_reshaped,
                        S2_grvi_med_reshaped,
                        S2_npci_med_reshaped,
                        S2_bg_med_reshaped])

# set up a list that stores the array names in corresponding order to the data stack
pred_names = [
    "bathymetry",                # water depth from BSH
    "sub_solidity",              # krigged substrate solidity
    "slope",                     # slope derived from BAW bathy
    "aspect",                    # aspect derived from BAW bathy
    "TPI",                       # tpi derived from BAW bathy
    "wave_velocity",              # bottom wave velocity derived from wind and bathymetry data (python script)
    "S2_blue",                   # blue band of single S2 scene
    "S2_green",                  # green band of single S2 scene
    "S2_red",                    # red band of single S2 scene   
    "S2_nir",                    # nir band of single S2 scene
    "S2_ndvi",                   # Normalized Difference Vegetation Index (NDVI)
    "S2_ndavi",                  # NDAVI - Normalized Difference Aquatic Vegetation Index
    "S2_wavi",                   # WAVI - Water Adjusted Vegetation Index
    "S2_grvi",                   # GRVI - Green-Red Vegetation Index
    "S2_npci",                   # Normalized Total Pigmet to Chl-a Ratio Index (NPCI)
    "S2_bgr",                    # Simple Ratio - BlueGreen
    "PS_blue",                   # blue band of single PS scene
    "PS_green",                  # green band of single PS scene
    "PS_red",                    # red band of single PS scene
    "PS_nir",                    # nir band of single PS scene
    "PS_ndvi",                   # Normalized Difference Vegetation Index (NDVI)
    "PS_ndavi",                  # NDAVI - Normalized Difference Aquatic Vegetation Index
    "PS_wavi",                   # WAVI - Water Adjusted Vegetation Index
    "PS_grvi",                   # GRVI - Green-Red Vegetation Index     
    "PS_npci",                   # Normalized Total Pigmet to Chl-a Ratio Index (NPCI)
    "PS_bgr",                    # Simple Ratio - BlueGreen
    "S2_blue_med",               # blue band of single S2 scene
    "S2_green_med",              # green band of single S2 scene
    "S2_red_med",                # red band of single S2 scene
    "S2_nir_med",                # nir band of single S2 scene
    "S2_ndvi_med",               # NDVI median from S-2 in GEE
    "S2_ndavi_med",              # NDAVI median from S-2 in GEE
    "S2_wavi_med",               # WAVI median from S-2 in GEE
    "S2_grvi_med",               # GRVI median from S-2 in GEE
    "S2_npci_med",               # NPCI median from S-2 in GEE
    "S2_bgr_med"]                # Simple Ratio - BlueGreen from S-2 in GEE

# # plot one of the bands/indices
# plt.imshow(np.squeeze(data_stack[pred_names.index("bathymetry")]), cmap='viridis', extent=[xmin, xmax, ymin, ymax], vmin=0, vmax=50)
# plt.title('raster values')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.colorbar(label='raster values')
# # Plot the GeoDataFrame as dots
# plt.scatter(kelp_gdf.geometry.x, kelp_gdf.geometry.y, color='red', s=3, label='Mapping sites')
# plt.show()

# write out the data stack as a GeoTIFF file
# get metadata from one of the input bands
meta = clipped_meta.copy()
meta.update({
    "count": len(pred_names),  # Number of bands
    "dtype": data_stack.dtype,  # Data type of the array
    "names": pred_names  # Assign band names
})
if output_mode :
    with rio.open(f"M://KELP_MA//data//raster_stacks//{date}_data_stack_kelp_modeling_36predictors.tif", 'w', **meta) as dst:
        for i, array_name in enumerate(pred_names):
            # write each band to the GeoTIFF file
            dst.write(data_stack[i], i + 1)  # band indexing starts from 1 in GeoTIFF
    print("Raster stack got successfully written out.")


#%%
##################################################################################################################
# Extract pixel values for the mapping sites #####################################################################
##################################################################################################################


# iterate over each point in the kelp GeoDataFrame
for index, point in kelp_gdf.iterrows():
    # get the coordinates of the point - based on GDf geometry
    x, y = point.geometry.x, point.geometry.y
    
    # convert the coordinates to pixel coordinates in the data stack
    col = int((x - clipped_meta['transform'][2]) / clipped_meta['transform'][0])  # column index
    row = int((y - clipped_meta['transform'][5]) / clipped_meta['transform'][4])  # row index
    
    # extract pixel values for each band
    for band_index, band_name in enumerate(pred_names):
        # extract pixel value for the band at the specified row and column
        pixel_value = data_stack[band_index, row, col]
        
        # add the pixel value as a new column in the kelp gdf
        kelp_gdf.at[index, band_name] = pixel_value


#%%
##################################################################################################################
# Correlation analysis ###########################################################################################
##################################################################################################################

"""
This section explores the relationship between the predictors and the target variables as well as 
between the predictors themself. Since the relationships between the predictors and the target variable are of 
biological manner it might be wise to use a non-linear correlation coefficient such as spearman to analyze them.
The correlation between the predictors themself (multicollinearity) should however be analyzed with a linear correlation
coefficient such as pearson. Due to the following Random Forest Regression, which is considered robust against multicollinearity
none of the predictors got dropped for the final model run (therefore it is commented out).
"""

# # compute Spearman correlation between each band and 'Lam_hyp'
# spearman_corr_results = kelp_gdf[pred_names].corrwith(kelp_gdf['Lam_hyp'], method='spearman')
# # create a DataFrame for correlation results
# spearman_corr_df = pd.DataFrame({'Correlation': spearman_corr_results})
# spearman_corr_df.reset_index(inplace=True)
# spearman_corr_df.rename(columns={'index': 'Bands'}, inplace=True)
# # calculate absolute correlation values
# spearman_corr_df['Absolute Correlation'] = abs(spearman_corr_df['Correlation'])
# # sort the DataFrame by the 'Correlation' column in descending order
# spearman_corr_df = spearman_corr_df.sort_values(by='Correlation', ascending=False).reset_index(drop=True)
# # generate a correlation plot using Spearman correlation (rotated by 90 degrees)
# plt.figure(figsize=(5, 10))
# plt.scatter(x=['Lam Hyp Cover'] * len(spearman_corr_df), y=spearman_corr_df['Bands'], 
#             s=500 * spearman_corr_df['Absolute Correlation'], c=spearman_corr_df['Correlation'], 
#             cmap='coolwarm', alpha=0.7, edgecolors='k')
# plt.title('Spearman correlation plot: \n Predictors vs Lam Hyp Cover')
# plt.yticks(rotation=0)
# cbar = plt.colorbar()
# cbar.set_label('Correlation')
# plt.grid(True)
# # add correlation values as text labels
# for i, txt in enumerate(spearman_corr_df['Correlation']):
#     plt.text(0.025, spearman_corr_df['Bands'][i], f'{txt:.2f}', ha='center', va='center', color='black', fontweight='bold')
# plt.tight_layout()
# plt.show()

# # define a thresold for spearman correlation with lam hyp cover; if the predictor doesn't reach the th it gets dropped
# #sp_th = 0.3
# sp_th = 0.0

# # filter out predictors with absolute correlation less than the threshold
# filtered_spearman_corr_df = spearman_corr_df[spearman_corr_df['Absolute Correlation'] >= sp_th].reset_index(drop=True)


# # generate a correlation plot using Spearman correlation 
# plt.figure(figsize=(5, 10))
# plt.scatter(x=['Lam Hyp Cover'] * len(filtered_spearman_corr_df), y=filtered_spearman_corr_df['Bands'], 
#             s=500 * filtered_spearman_corr_df['Absolute Correlation'], c=filtered_spearman_corr_df['Correlation'], 
#             cmap='coolwarm', alpha=0.7, edgecolors='k')
# plt.title('Spearman correlation plot: \n Predictors vs Lam Hyp Cover')
# plt.yticks(rotation=0)
# cbar = plt.colorbar()
# cbar.set_label('Correlation')
# plt.grid(True)
# # add correlation values as text labels
# for i, txt in enumerate(filtered_spearman_corr_df['Correlation']):
#     plt.text(0.025, filtered_spearman_corr_df['Bands'][i], f'{txt:.2f}', ha='center', va='center', color='black', fontweight='bold')
# plt.tight_layout()
# plt.show()

# # create list of correlated predictors
# preds_corr = list(filtered_spearman_corr_df['Bands'])


# # generate a corr matrix between all predictor variables with pearson
# corr_matrix = kelp_gdf[preds_corr].corr(method='pearson')
# # Create a mask for the upper triangle
# mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
# # Set up the matplotlib figure
# plt.figure(figsize=(10, 8))
# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", square=True, 
#             linewidths=.5, cbar_kws={"shrink": .75, "orientation": "horizontal", "pad": -1.25})
# plt.title('Pearson correlation matrix of important predictors')
# plt.tight_layout()
# plt.show()

# # drop one predictor of each predictor pair that shows high pearson correlation (> 0.9)
# # to reduce redundancy and increase interpretability
# # preds_to_remove = ['PS_blue', 'PS_gndvi', 'S2_ndti', 'S2_gndvi', 'S2series_gndvi_med']
# preds_to_remove = []
# preds_clean = [pred for pred in preds_corr if pred not in preds_to_remove]

# # generate a corr matrix between all predictor variables with pearson
# corr_matrix = kelp_gdf[preds_clean].corr(method='pearson')
# # Create a mask for the upper triangle
# mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
# # Set up the matplotlib figure
# plt.figure(figsize=(10, 8))
# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", square=True, 
#             linewidths=.5, cbar_kws={"shrink": .75, "orientation": "horizontal", "pad": -1.25})
# plt.title('Pearson correlation matrix of important predictors - cleaned')
# plt.tight_layout()
# plt.show()


#%%
##################################################################################################################
# Preparing the spatial model ####################################################################################
##################################################################################################################

"""
This section prepares the field data for modeling. Sampling sites that show a high deviation (>= 1.77 m) between 
measured depth by the boat's echosounder and water depth derived from the bathymetric map are exlucded from the
modeling process. The remaining field data gets split into a training set and a test set in stratified manner
(70-30 split).
"""

### adjust the kelp gdf ###
kelp_gdf = kelp_gdf.rename(columns={"solid_subs_all_considered30": "solid_substrate"})
# only keep cols that are of further interest to optimize readability
cols_to_keep = ['Lam_hyp', 'Remarks', 'Depth_corrected_m']
cols_to_keep.extend(pred_names) # add all predictors that were generated at the beginngin
kelp_gdf = kelp_gdf[cols_to_keep]

# ### subsetting the kelp gdf due to insecure cover evaluations (not done for final model)
# kelp_gdf_secure = kelp_gdf[kelp_gdf['Remarks'] != 'unsecure']
# kelp_gdf = kelp_gdf_secure

# ### subsetting the kelp gdf due to minimum depth (not done for final model)
# kelp_gdf_shallow = kelp_gdf[kelp_gdf['Depth_corrected_m'] <= 6]
# kelp_gdf = kelp_gdf_shallow

### subsetting the kelp gdf due to depth deviations ###
# scatter plot between depth from echosounder and bathymetry map from BSH
plt.figure(figsize=(8, 6))
plt.scatter(kelp_gdf['Depth_corrected_m'], kelp_gdf['bathymetry'], alpha=0.5)
# fit a linear regression model
model = LinearRegression()
X = kelp_gdf['Depth_corrected_m'].values.reshape(-1, 1)
y = kelp_gdf['bathymetry'].values.reshape(-1, 1)
model.fit(X, y)
# predict y values
y_pred = model.predict(X)
# calculate R² and standard deviation
r_squared = r2_score(y, y_pred)
std_dev = np.std(y - y_pred)
no_obs = len(kelp_gdf)
# plot the regression line
plt.plot(X, y_pred, color='red', linestyle='--', label=f'Regression Line (R² = {r_squared:.2f}, N = {no_obs}')
plt.xlim(-0.5, 16)
plt.ylim(-0.5, 16)
# add labels and title
plt.xlabel('Depth_corrected_m')
plt.ylabel('Water depth based on boats echosounder')
plt.title('Scatter Plot: Depth_corrected_m vs bathymetry')
# show the plot
plt.legend()
plt.grid(True)
plt.show()
# compute the absolute difference between 'bathymetry' and 'Depth_corrected_m'
depth_diff = kelp_gdf['Depth_corrected_m'] - kelp_gdf['bathymetry']
depth_diff_abs = np.abs(kelp_gdf['Depth_corrected_m'] - kelp_gdf['bathymetry'])
# compute standard deviation of the absolute depth differences
std_dev_diff = np.std(np.abs(depth_diff))
# create a histogram with KDE curve
plt.figure(figsize=(8, 6))
sns.histplot(depth_diff, bins=25, kde=True, color='blue', edgecolor='black', alpha=0.8, zorder=3)
# plot marks for 1st, 2nd, and 3rd standard deviations
for i in range(-3, 4):
    if i in [-3, 3]:
        plt.axvline(i * std_dev_diff, color='red', linestyle='--', linewidth=1)
plt.xlabel('Depth Difference [measured vs. bathymetric map]', fontsize=14)
plt.ylabel('Number of Sites', fontsize=14)
plt.title(f'Frequency Distribution of Depth Differences (σ = {std_dev_diff:.2f})', fontsize=14)
plt.grid(True, zorder=0)
plt.show()
# drop outliers in the kelp gdf which havea large difference in depth  (>= 3*std deviation)
depth_th = 3 * std_dev_diff # leads to a drop of 30 sites
kelp_gdf_depth_corrected = kelp_gdf[depth_diff_abs <= depth_th]

# create features and target variable for 2023
X = kelp_gdf_depth_corrected[pred_names] # predictor variables
y = kelp_gdf_depth_corrected['Lam_hyp'] # target variable

# define random state in case reproducable result is desired
random_state = 188
# split the data into training and validation set; it is important to include stratification here due to small classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y) 

#%%
##################################################################################################################
# Hyperparameter tuning using Cross-Validation (prior feature selection) #########################################
##################################################################################################################

"""
This section conducts hyperparameter tuning based on Optuna optimization for an RFR using all 36 predictors. The 
optimized set of hyperparameters is then used in every subsequent modeling loop to ensure comparability between
the different model performances.
"""

# def objective(trial):
#     # define hyperparameter search space
#     n_estimators = trial.suggest_int('n_estimators', 100, 500)
#     max_depth = trial.suggest_int('max_depth', 5, 20)
#     min_samples_split = trial.suggest_int('min_samples_split', 5, 20)
#     min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 10)
#     bootstrap = trial.suggest_categorical('bootstrap', [True]) # generally true  for RF
    
#     # create the model with suggested hyperparameters
#     rfr = RandomForestRegressor(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         min_samples_split=min_samples_split,
#         min_samples_leaf=min_samples_leaf,
#         bootstrap=bootstrap,
#         random_state=random_state  # Ensure reproducibility
#     )
    
#     # perform cross-validation and calculate the mean R² score
#     scores = cross_val_score(rfr, X_train, y_train, cv=10, scoring='r2')
#     mean_r2_score = scores.mean()
    
#     return mean_r2_score

# # create the Optuna study
# study = optuna.create_study(direction='maximize')

# # run the optimization with parallel jobs and time limit
# study.optimize(objective, n_trials=100, n_jobs=-1, timeout=3600)  # adjust n_trials and timeout as needed

# # get the best parameters and score
# best_params = study.best_params # these are the hyperparameters that will get used in following feature selection loop
# best_score = study.best_value
# print(f"Best hyperparameters: {best_params}")
# print(f"Best R² Score: {best_score}")

# "best_params" stores the optimized hyperparameters so that a reevaluation isn't required each time the script is run
best_params =  {'n_estimators': 469, 'max_depth': 6, 'min_samples_split': 5, 'min_samples_leaf': 2, 'bootstrap': True} # for 36F, RS 188


#%%
##################################################################################################################
# Selecting the model with optimal number of features ############################################################
##################################################################################################################

"""
In this section, the different predictor sets of interest are defined and the model evaluation loop is run for each
of them. This way, the best-performing feature combination (in the RFR) for each predictor set is identified. Accuracy 
metrics and regression plots are generated to evaluate model performances and SHAP values and SHAP summary plots are 
generated to evaluate feature importances. 
"""

### define different sets of predictors for the different runs ###
# prepare run utilizing non-RS predictors only
X_train_no_RS = X_train.copy()
X_test_no_RS = X_test.copy()
X_train_no_RS = X_train_no_RS.drop(X_train_no_RS.filter(regex="S2|PS").columns,axis=1)
X_test_no_RS = X_test_no_RS.drop(X_test_no_RS.filter(regex="S2|PS").columns,axis=1)
# prepare run utilizing S-2 monotemporal predictors only
X_train_S2 = X_train.copy()
X_test_S2 = X_test.copy()
X_train_S2 = X_train_S2.drop(X_train_S2.filter(regex="med|PS|bathy|solidity|slope|aspect|TPI|wave").columns,axis=1)
X_test_S2 = X_test_S2.drop(X_test_S2.filter(regex="med|PS|bathy|solidity|slope|aspect|TPI|wave").columns,axis=1)
# prepare run utilizing PS monotemporal predictors only
X_train_PS = X_train.copy()
X_test_PS = X_test.copy()
X_train_PS = X_train_PS.drop(X_train_PS.filter(regex="S2|bathy|solidity|slope|aspect|TPI|wave").columns,axis=1)
X_test_PS = X_test_PS.drop(X_test_PS.filter(regex="S2|bathy|solidity|slope|aspect|TPI|wave").columns,axis=1)
# prepare run utlizing S-2 time series predictors only 
# "S2(?!.*_med)" pattern matches any string that starts with "S2" but does not end with "_med"
X_train_S2series = X_train.copy()
X_test_S2series = X_test.copy()
X_train_S2series = X_train_S2series.drop(X_train_S2series.filter(regex="S2(?!.*_med)|PS|bathy|solidity|slope|aspect|TPI|wave").columns,axis=1) 
X_test_S2series = X_test_S2series.drop(X_test_S2series.filter(regex="S2(?!.*_med)|PS|bathy|solidity|slope|aspect|TPI|wave").columns,axis=1)
# prepare run utilizing all predictors combined
X_train_all = X_train.copy()
X_test_all = X_test.copy()

# prepare run utilizing non-RS predictors + mono S-2 predictors only
X_train_noRS_S2 = X_train.copy()
X_test_noRS_S2 = X_test.copy()
X_train_noRS_S2 = X_train_noRS_S2.drop(X_train_noRS_S2.filter(regex="med|PS").columns,axis=1)
X_test_noRS_S2 = X_test_noRS_S2.drop(X_test_noRS_S2.filter(regex="med|PS").columns,axis=1)
# prepare run utilizing non-RS predictors + mono S-2 predictors only
X_train_noRS_PS = X_train.copy()
X_test_noRS_PS = X_test.copy()
X_train_noRS_PS = X_train_noRS_PS.drop(X_train_noRS_PS.filter(regex="S2").columns,axis=1)
X_test_noRS_PS = X_test_noRS_PS.drop(X_test_noRS_PS.filter(regex="S2").columns,axis=1)
# prepare run utilizing non-RS predictors + mono S-2 predictors only
X_train_noRS_S2series = X_train.copy()
X_test_noRS_S2series = X_test.copy()
X_train_noRS_S2series = X_train_noRS_S2series.drop(X_train_noRS_S2series.filter(regex="PS|S2(?!.*_med)").columns,axis=1)
X_test_noRS_S2series = X_test_noRS_S2series.drop(X_test_noRS_S2series.filter(regex="PS|S2(?!.*_med)").columns,axis=1)


# define a function that finds the optimal number of predictors for the given predictor subset based on SHAP values and R²
def choose_final_model(X_train, X_test, params, run_setting):
    
    # store current run setting as string
    curr_setting = run_setting
    
    # set initial set of trainingsdata that includes all features and store starting number of features
    curr_X_train = X_train
    curr_X_test = X_test
    start_feat_count = len(X_train.columns)
    # initialize variable that will stores the best r² score and its feature selection
    r2_best = -9999
    r2_list = []
    rmse_list = []
    featcount_list = []
    
    # intitialize a loop to find the best number of features based on RMSE value
    for i in range(0, start_feat_count+1):
        
        curr_feature_count = start_feat_count - i
        # break the loop if feature count reaches 0
        if curr_feature_count == 0:
            print("No features left for evaluation.")
            break
        
        print("############################################")
        print("Current feature count: ", curr_feature_count)
        
        # initialize the RFR with the pre-evaluated set of hyperparameters
        rf_regressor = RandomForestRegressor(**best_params, random_state=random_state)
        rf_regressor.fit(curr_X_train, y_train)
        
        # compute SHAP values with additivity check disabled
        explainer = shap.TreeExplainer(rf_regressor, check_additivity=False)
        shap_values = explainer.shap_values(curr_X_train)
        
        # get feature importance rankings based on SHAP values
        shap_importances = np.abs(shap_values).mean(axis=0)
        
        # get summary plots for the current model
        shap.summary_plot(shap_values, curr_X_train, show=False, max_display=curr_feature_count)
        fig = plt.gcf()
        ax = plt.gca()
        ax.set_xlim(-50, 50)  # set fixed x-axis scale
        feature_names = [tick.get_text() for tick in ax.get_yticklabels()]
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        shap_dict = dict(zip(curr_X_train.columns, mean_shap_values))
        for j, feature in enumerate(feature_names):
            value = shap_dict.get(feature, 0)
            ax.text(44, j, f'Ø {value:.2f}', va='center', ha='left', color='black', fontsize=10,
                    path_effects=[pe.withStroke(linewidth=1, foreground="white")])
        plt.title(f"{curr_setting} RFR Run using {curr_feature_count} Features", fontsize=14)
        ax.set_xlabel("SHAP Value (Impact on predicted $\it{{L.\ hyperborea}}$ Cover [%])", fontsize=14)
        cbar = plt.gcf().axes[-1]  # Access the colorbar, which is the last axis in the figure
        cbar.tick_params(labelsize=14)  # Set the desired label size here
        plt.tight_layout()
        # save shap summary plot
        if output_mode == True:
            plt.savefig(rf"M:\KELP_MA\results\final\model_runs\{curr_setting}\{curr_setting}_RFR_shap_summary_{curr_feature_count}-features", dpi=300, bbox_inches='tight')
        plt.show()
        
        # predict with the current selection of features
        curr_y_pred = rf_regressor.predict(curr_X_test)

        # evaluate the model with the current selection of features
        current_r2 = r2_score(y_test, curr_y_pred)
        
        # calculate adjusted R² --> necessary for comparing the performance of models with changing number of predictors
        n = len(y_test)  # number of observations
        p = curr_feature_count  # number of predictors
        current_r2_adj = 1 - (1 - current_r2) * (n - 1) / (n - p - 1)
                     
        # scatterplot 2D histogram (credits to Felix Linhardt (CAU Kiel) who helped me to generate this plot)
        # define bin edges
        bingrenzen = [-5 + i * 10 for i in range(12)]
        # 2D histogram with logarithmic normalization
        hist, xedges, yedges, img = plt.hist2d(y_test, curr_y_pred, bins=bingrenzen, cmap="Greens", 
                                               norm=mcolors.LogNorm(vmin=1, vmax=60))  # vmin and vmax define the normalization range
        cbar = plt.colorbar(img)
        cbar.set_ticks([1, 10, 60])  # manually set the ticks for the log scale
        cbar.set_ticklabels(['1', '10', '60'])  # label them appropriately
        cbar.ax.set_ylabel('Number of Points', rotation=90)
        plt.scatter(y_test, curr_y_pred, marker=".", color="#777777", s=2)
        plt.xticks([i * 10 for i in range(11)])
        plt.yticks([i * 10 for i in range(11)])
        plt.xlabel('Observed Cover of $\it{{L.\ hyperborea}}$ [%]')
        plt.ylabel('Predicted Cover of $\it{{L.\ hyperborea}}$ [%]')
        plt.xlim((-5, 105))
        plt.ylim((-5, 105))        
        # add regression line
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_test, curr_y_pred)
        plt.axline(xy1=(0, intercept), slope=slope, linestyle="--", color="r", 
                   label=f'$y = {slope:.1f}x {intercept:+.1f}, r={r_value:.1f}$')
        plt.plot([-5, 105], [-5, 105], color='black', linestyle='--')      
        plt.legend(loc='upper left')
        curr_rmse = np.sqrt(mean_squared_error(y_test, curr_y_pred))
        n_obs = len(y_test)      
        plt.title(f"{curr_setting} RFR Run using {curr_feature_count} Features\n R²: {round(current_r2, 3)}, R² (adj.): {round(current_r2_adj, 3)}, RMSE: {round(curr_rmse, 2)}, n = {n_obs}", fontsize=12)
        plt.tight_layout()
        if output_mode == True:
            plt.savefig(rf"M:\KELP_MA\results\final\model_runs\{curr_setting}\{curr_setting}_RFR_scatter_{curr_feature_count}_features", dpi=300, bbox_inches='tight')
        plt.show()
        
        # update best R² and its features if they outperform the previous model runs
        if current_r2 > r2_best:
            r2_best = current_r2
            best_features = curr_X_train.columns
            best_features_count = len(best_features)
        
        # get name of feature with the lowest impact and drop it from the X_train dataframe
        shap_importance_list = shap_importances.tolist()
        drop_index = shap_importance_list.index(min(shap_importance_list))
        drop_name = curr_X_train.columns[drop_index]
        print(f"Feature {drop_name} showed the lowest SHAP importance and got dropped")
        curr_X_train = curr_X_train.drop(curr_X_train.columns[drop_index], axis=1) # drop the feature
        curr_X_test = curr_X_test.drop(curr_X_test.columns[drop_index], axis=1)
        
        # store current R², RMSE and feature count in a list
        r2_list.append(current_r2)
        rmse_list.append(curr_rmse)
        featcount_list.append(curr_feature_count)
    
    
    print(f"Best performing number of features in RFR is {best_features_count} with R² of {round(r2_best,2)}")
        
    # create a figure and a set of subplots --> include into function above
    fig, ax1 = plt.subplots()
    # plot the R² values on the first y-axis
    ax1.set_xlabel('Number of Features used in RFR')
    ax1.set_ylabel('R²', color='tab:blue')
    ax1.plot(featcount_list, r2_list, color='tab:blue', label='R²')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    # create a second y-axis to plot the RMSE values
    ax2 = ax1.twinx()
    ax2.set_ylabel('RMSE', color='tab:green')
    ax2.plot(featcount_list, rmse_list, color='tab:green', label='RMSE')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    plt.title(f'{curr_setting} RFR Runs: R² and RMSE based on Feature Count', fontsize=14)
    fig.tight_layout()  
    # save R²-RMSE-feature graph
    if output_mode == True:
        plt.savefig(rf"M:\KELP_MA\results\final\model_runs\{curr_setting}\{curr_setting}_accuracy_based_on_fcount", dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_features, r2_best, r2_list, rmse_list, featcount_list
    
### run the feature evaluation and selection process for the different settings ###  
best_features_no_RS, r2_best_no_RS, r2_list_no_RS, rmse_list_no_RS, featcount_list_no_RS = choose_final_model(X_train=X_train_no_RS, X_test=X_test_no_RS, params=best_params, run_setting="Non_RS")   
best_features_PS, r2_best_PS, r2_list_PS, rmse_list_PS, featcount_list_PS = choose_final_model(X_train=X_train_PS, X_test=X_test_PS, params=best_params, run_setting="PS_Mono")   
best_features_S2, r2_best_S2, r2_list_S2, rmse_list_S2, featcount_list_S2 = choose_final_model(X_train=X_train_S2, X_test=X_test_S2, params=best_params, run_setting="S2_Mono")    
best_features_S2series, r2_best_S2series, r2_list_S2series, rmse_list_S2series, featcount_list_S2series = choose_final_model(X_train=X_train_S2series, X_test=X_test_S2series, params=best_params, run_setting="S2_Series")    
best_features_noRS_PS, r2_best_noRS_PS, r2_list_noRS_PS, rmse_list_noRS_PS, featcount_list_noRS_PS = choose_final_model(X_train=X_train_noRS_PS, X_test=X_test_noRS_PS, params=best_params, run_setting="Non_RS+PS_Mono")  
best_features_noRS_S2, r2_best_noRS_S2, r2_list_noRS_S2, rmse_list_noRS_S2, featcount_list_noRS_S2 = choose_final_model(X_train=X_train_noRS_S2, X_test=X_test_noRS_S2, params=best_params, run_setting="Non_RS+S2_Mono")    
best_features_noRS_S2series, r2_best_noRS_S2series, r2_list_noRS_S2series, rmse_list_noRS_S2series, featcount_list_noRS_S2series = choose_final_model(X_train=X_train_noRS_S2series, X_test=X_test_noRS_S2series, params=best_params, run_setting="Non_RS+S2_series")   
best_features_all, r2_best_all, r2_list_all, rmse_list_all, featcount_list_all = choose_final_model(X_train=X_train_all, X_test=X_test_all, params=best_params, run_setting="All_combined")   

#%%
##################################################################################################################
# Raster predictions with optimal number of features and area computation ########################################
##################################################################################################################

"""
This section generates raster predictions for each of the best-performing models of the previous section, illustrates
the spatial predictions as a map, and transforms the L. hyperborea cover in percent into actual area estimates.
"""

### define different sets of predictors for the different runs ###
# extract predictor indices for each different setting
pred_names_no_RS_ind = [i for i, name in enumerate(pred_names) if name in list(best_features_no_RS)]
pred_names_PS_ind = [i for i, name in enumerate(pred_names) if name in list(best_features_PS)]
pred_names_S2_ind = [i for i, name in enumerate(pred_names) if name in list(best_features_S2)]
pred_names_S2series_ind = [i for i, name in enumerate(pred_names) if name in list(best_features_S2series)]
pred_names_noRS_PS_ind = [i for i, name in enumerate(pred_names) if name in list(best_features_noRS_PS)]
pred_names_noRS_S2_ind = [i for i, name in enumerate(pred_names) if name in list(best_features_noRS_S2)]
pred_names_noRS_S2series_ind = [i for i, name in enumerate(pred_names) if name in list(best_features_noRS_S2series)]
pred_names_all_ind = [i for i, name in enumerate(pred_names) if name in list(best_features_all)]

# flatten each array of interest in the data stack to 1D to prepare spatial predictions for each setting
ds_flat_no_RS = [data_stack[i].flatten() for i in pred_names_no_RS_ind]
ds_flat_PS = [data_stack[i].flatten() for i in pred_names_PS_ind]
ds_flat_S2 = [data_stack[i].flatten() for i in pred_names_S2_ind]
ds_flat_S2series = [data_stack[i].flatten() for i in pred_names_S2series_ind]
ds_flat_noRS_PS = [data_stack[i].flatten() for i in pred_names_noRS_PS_ind]
ds_flat_noRS_S2 = [data_stack[i].flatten() for i in pred_names_noRS_S2_ind]
ds_flat_noRS_S2series = [data_stack[i].flatten() for i in pred_names_noRS_S2series_ind]
ds_flat_all = [data_stack[i].flatten() for i in pred_names_all_ind]
# create a design matrix, which is a 2D array, created from all 1D arrays for each setting
X_raster_no_RS = np.column_stack(ds_flat_no_RS)
X_raster_PS = np.column_stack(ds_flat_PS)
X_raster_S2 = np.column_stack(ds_flat_S2)
X_raster_S2series = np.column_stack(ds_flat_S2series)
X_raster_noRS_PS = np.column_stack(ds_flat_noRS_PS)
X_raster_noRS_S2 = np.column_stack(ds_flat_noRS_S2)
X_raster_noRS_S2series = np.column_stack(ds_flat_noRS_S2series)
X_raster_all = np.column_stack(ds_flat_all)

# prepare the bathymetry mask for clipping out areas that are not of interest
# mask out very deep water and non water areas
bathy_flat = np.squeeze(bathymetry_data, axis=0)
depth_mask = ((bathy_flat < -12.5) | (bathy_flat > 0))


def generate_raster_prediction(X_raster_final, X_train_final, params, run_setting):

    # initialize the RFR with the pre-evaluated set of hyperparameters and fit trainingdata for current setting
    rf_regressor = RandomForestRegressor(**params, random_state=random_state)
    rf_regressor.fit(X_train_final, y_train)
    
    # generate raster predictions with the trained RFR and the respective features
    predicted_raster_values = rf_regressor.predict(X_raster_final)

    # reshape the predicted values back into the original raster dimensions
    rf_raster_pred = predicted_raster_values.reshape(clipped_meta['height']-1, clipped_meta['width']-1)

    # mask out very deep water and non water areas
    rf_raster_pred = np.where(depth_mask, np.nan, rf_raster_pred)

    # mask out pixels with very low prediction of lam hyp cover (< 5 %)
    rf_raster_pred[rf_raster_pred < 5] = np.nan
    
    final_fcount = len(X_train_final.columns)
    
    ### create nice-looking map of the predicted L. hyperborea cover ### 
    # plot raster predictions
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    plt.imshow(rf_raster_pred, extent=[bathymetry_extent.left, bathymetry_extent.right, bathymetry_extent.bottom, bathymetry_extent.top], cmap='Greens', vmin=0, vmax=100)
    cbar = plt.colorbar(label='Predicted Cover of $\it{{L.\ hyperborea}}$  [%]')
    cbar.set_label(label='Predicted Cover of $\it{{L.\ hyperborea}}$ [%]', size=16)
    # plot the Helgoland polygon
    helgoland_poly = gpd.read_file(helgoland_polygon_path)
    helgoland_poly.plot(ax=plt.gca(), facecolor='#D2B48C', edgecolor='black', linewidth=0.11)
    # define custom colormap with three shades of blue for the bathy isolines
    colors = ['#002060', '#0070C0', '#00B0F0']
    cmap = mcolors.ListedColormap(colors)
    # plot isolines with custom colormap
    isolines_filtered.plot(ax=plt.gca(), column='ELEV', cmap=cmap, linewidth=0.5, legend=False)
    # create custom legend for water depth classes
    legend_labels = {'5': colors[2], '10': colors[1], '15': colors[0]}
    legend_lines = [Line2D([0], [0], color=color, linewidth=3, linestyle='-') for _, color in legend_labels.items()]
    plt.legend(legend_lines, 
               legend_labels.keys(), 
               loc='upper right', 
               bbox_to_anchor=(1, 1), 
               title='Bathymetric\n Isolines [m]', 
               fontsize='large', 
               markerscale=2,  
               title_fontsize='large',  
               #borderpad=1,             
               labelspacing=0.75)
    # annotate polygons with the values from the column 'Gebiet'
    for x, y, label_text in zip(helgo_poly.geometry.centroid.x, helgo_poly.geometry.centroid.y, helgo_poly["Gebiet"]):
        ax.annotate(label_text, (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)
    # add CRS information as text in the bottom left corner
    crs_text = "CRS: WGS 84 / UTM zone 32N"  # extract CRS information from the raster
    plt.text(0.01, 0.005, crs_text, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='left')
    # add title and labels
    plt.title(f'Prediction Map of the {run_setting} RFR using {final_fcount} Features ', fontsize = 16)
    plt.xlabel('Easting', fontsize = 16)
    plt.ylabel('Northing', fontsize = 16)
    # adjust tick size for map and colorbar
    ax.tick_params(axis='both', which='major', labelsize=12, length=6)
    cbar.ax.tick_params(labelsize=12, length=6)
    
    # save plots and raster
    if output_mode == True:
        plt.savefig(rf"M:\KELP_MA\results\final\model_runs\{run_setting}\RFR_pred_lam_hyp_raster", dpi=300, bbox_inches='tight')
        with rasterio.open(rf"M:\KELP_MA\results\final\model_runs\{run_setting}\RFR_pred_lam_hyp_raster.tif", 'w', **target_profile) as dst:
            dst.write(rf_raster_pred, 1)
    
    plt.tight_layout()
    plt.show()
    
    ## set pixel size for area computation
    pixel_area_m2 = 25
    # compute area of predicted lam hyp cover in km² (area is already masked by depth and small predictions)
    lh_area_km2 = round(np.nansum(rf_raster_pred*pixel_area_m2/100)/1000000,2)
    print(f"The most accuracte RFR of setting {run_setting} predicts {lh_area_km2} km² of L. hyperborea throughtout the AOI")
    

# raster prediction for "no_RS" (no Remote Sensing data)
generate_raster_prediction(X_raster_final=X_raster_no_RS, 
                           X_train_final=X_train_no_RS[best_features_no_RS], 
                           params=best_params, 
                           run_setting="Non_RS")

# raster prediction for "PS_mono" (PlanetScope single scene)
generate_raster_prediction(X_raster_final=X_raster_PS, 
                           X_train_final=X_train_PS[best_features_PS], 
                           params=best_params, 
                           run_setting="PS_Mono")

# raster prediction for "S2_mono" (Sentinel-2 single scene)
generate_raster_prediction(X_raster_final=X_raster_S2, 
                           X_train_final=X_train_S2[best_features_S2], 
                           params=best_params, 
                           run_setting="S2_Mono")

# raster prediction for "S2_series" (Sentinel-2 time series)
generate_raster_prediction(X_raster_final=X_raster_S2series, 
                           X_train_final=X_train_S2series[best_features_S2series], 
                           params=best_params, 
                           run_setting="S2_Series")

# raster prediction for Non_RS predictors combined with PS_Mono predictors
generate_raster_prediction(X_raster_final=X_raster_noRS_PS, 
                           X_train_final=X_train_noRS_PS[best_features_noRS_PS], 
                           params=best_params, 
                           run_setting="Non_RS+PS_Mono")

# raster prediction for Non_RS predictors combined with S2_Mono predictors
generate_raster_prediction(X_raster_final=X_raster_noRS_S2, 
                           X_train_final=X_train_noRS_S2[best_features_noRS_S2], 
                           params=best_params, 
                           run_setting="Non_RS+S2_Mono")

# raster prediction for Non_RS predictors combined with S2_Series predictors
generate_raster_prediction(X_raster_final=X_raster_noRS_S2series, 
                           X_train_final=X_train_noRS_S2series[best_features_noRS_S2series], 
                           params=best_params, 
                           run_setting="Non_RS+S2_series")

# raster prediction for "All_combined" (All features combined)
generate_raster_prediction(X_raster_final=X_raster_all, 
                           X_train_final=X_train_all[best_features_all], 
                           params=best_params, 
                           run_setting="All_combined")

#%%

