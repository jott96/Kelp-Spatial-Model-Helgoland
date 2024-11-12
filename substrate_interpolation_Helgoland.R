####################################################################################
# This script performs ordinary kriging to estimate substrate solidity at the seabed
# of the coastal waters around the island of Helgoland. The binary substrate solidity
# variable is derived from the field data of the kelp monitoring around the island in
# September 2023. Loose substrates are encoded with 0, while solid substrates are 
# encoded with 1.
#####################################################################################

# load required packages
library(sp)
library(sf)
library(gstat)
library(automap)
library(spdep)
library(raster)
library(ggplot2)
library(geoR)
library(automap)

# set path to specific datasets
m_table_path = "C:\\Helgoland_Kelp_Project\\GIS_data\\test_data\\24-02-04_kelp_master_table.gpkg"
bathy_path = "C:\\Helgoland_Kelp_Project\\GIS_data\\AOI\\24-04-29_Bathy_MLWS_5m_raster_aoi_clipped.tif" # used for getting the spatial extent of the aoi and creating the target raster

#################################
## data preparation #############
#################################

# load the geopackage of the mapping table
kelp_data = st_read(m_table_path)

# convert data into a df
kelp_df = as.data.frame(kelp_data)

# convert sf object to SpatialPointsDataFrame
kelp_spdf = as(kelp_data, "Spatial")

# print the resulting SpatialPointsDataFrame
print(kelp_spdf)

# load the raster file for grid generation
bathy_raster = raster(bathy_path)

# generate a spatial grid from the bathy raster
target_raster = raster(bathy_raster)
res(target_raster) = 5 # set spatial resolution to 5m
kelp_grid = as(target_raster, 'SpatialGrid')

### check for spatial autocorrelation before kriging ###
# create spatial weights matrix
coords = coordinates(kelp_spdf)
nb = knn2nb(knearneigh(coords, k = 4)) # k-nearest neighbors with k = 4
# convert neighbors to spatial weights
sw = nb2listw(nb, style = "W")  # W for row-standardized weights
# compute Moran's I statistic for spatial autocorrelation of substrate column
morans = moran.test(kelp_spdf$solid_susb_all_considered30, listw = sw)
print(morans)

#################################
## ordinary kriging #############
#################################

# generate a variogram cloud (not effective for binary variable)
#vcloud = variogram(kelp_spdf$solid_susb_all_considered30~1, locations=kelp_spdf, width=100, cloud = TRUE)
#plot(vcloud)

# generate a sample variogram 
s_vario = variogram(kelp_spdf$solid_susb_all_considered30~1, locations=kelp_spdf, width=100)
plot(s_vario) # estimated values: sill - 1000, range - 1550, nugget - 900

# explore mathematical functions for the estimation
vgm()

# fit the variogram
f_vario = fit.variogram(s_vario, model = vgm(psill = 0.23, model = "Exp", range = 1200, nugget = 0.13))
plot(f_vario, cutoff=3000)

# plot sample- and fitted variogram - how well was the fit?
plot(s_vario, f_vario, main = 'Exponential variogram model for SSI', xlab = "Distance [m]")


#################################
## LOOCV ########################
#################################

# perform leave-one-out cross-validation
cv_results = krige.cv(formula = solid_susb_all_considered30 ~ 1, locations = kelp_spdf, model = f_vario, nmax = 30)

# print the cross-validation results
print(cv_results)

# plot the observed vs. predicted values
ggplot(as.data.frame(cv_results), aes(x = observed, y = var1.pred)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Observed vs. Predicted (LLOCV)", x = "Observed", y = "Predicted")

# calculate and print the Mean Error (ME)
mean_error = mean(cv_results$var1.pred - cv_results$observed)
cat("Mean Error (ME): ", mean_error, "\n")

# calculate and print the Mean Squared Error (MSE)
mean_squared_error = mean((cv_results$var1.pred - cv_results$observed)^2)
cat("Mean Squared Error (MSE): ", mean_squared_error, "\n")

# calculate and print the Root Mean Squared Error (RMSE)
rmse = sqrt(mean_squared_error)
cat("Root Mean Squared Error (RMSE): ", rmse, "\n")

# calculate and plot residuals
residuals = cv_results$observed - cv_results$var1.pred

# plot residuals
ggplot(as.data.frame(cv_results), aes(x = observed, y = residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Residuals (LLOCV)", x = "Observed", y = "Residuals")

#################################
## Final Kriging Prediction #####
#################################

# perform the kriging operation with the sample data and the variogram model
kriging_result =  gstat(formula = kelp_spdf$solid_susb_all_considered30~1, locations=kelp_spdf, model=f_vario)

# perform the prediction on the grid
kriged_map = predict(kriging_result, kelp_grid)
plot(kriged_map)

max(kriged_map$var1.pred)
min(kriged_map$var1.pred)

# export raster as GeoTIFF
kriged_raster = raster(kriged_map)
#writeRaster(kriged_raster, "C:\\Helgoland_Kelp_Project\\GIS_data\\test_data\\test_results\\24-13-05_pred_subs_OK_5x5m_all_considered30.tif", format = "GTiff", overwrite = TRUE)





