# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 13:07:11 2024

@author: Jannes Ott
"""
# list of PlanetScope band names (loop up in documentation)
# =============================================================================
# Coastal Blue: 431 - 452 nm
# Blue: 465 – 515 nm
# Green I: 513 - 549 nm
# Green: 547 – 583 nm
# Yellow: 600 - 620 nm
# Red: 650 – 680 nm
# RedEdge: 697 – 713 nm
# NIR: 845 – 885 nm
# =============================================================================
#PS_band_names = ["coastal_blue", "blue", "green_i", "green", "yellow", "red", "rededge", "nir"]

###############################################################################
###############################################################################
###############################################################################

import os
import glob
import rasterio
from rasterio.mask import mask
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import geopandas as gpd
from shapely.geometry import mapping
import matplotlib.pyplot as plt

# Load the geopackage with field data
field_data = gpd.read_file(r"M:\KELP_MA\data\field_data\23-11-05_kelp_2023_test_data_refac_32632.gpkg")

# this is a path towards the iamge folder
PS_data_path = "M://KELP_MA//data//RS_data//PlanetScope//KELP_2023_psscene_analytic_8b_sr_udm2//PSScene//"

# turn all percentual coverage values into a 0/1 codation for absence/abundace based on a threshold
coverage_th = 10 
field_data['Lam_hyp'] = field_data['Lam_hyp'].apply(lambda x: 0 if x <= coverage_th else 1)

# get a list with all the .tif files 
file_extension = "*SR_8b_clip.tif"
search_pattern = os.path.join(PS_data_path , file_extension)
PS_imagery_list = glob.glob(search_pattern)
print(PS_imagery_list)

with rasterio.open(PS_imagery_list[0]) as src:
    date = src.name[82:90]

# Extract training samples using field data
training_data = []
for index, row in field_data.iterrows():
    # Load the PlanetScope tif file for each iteration
    tif_file = 'path/to/your/tif/file.tif'
    with rasterio.open(PS_imagery_list[0]) as src:
        geom = row['geometry']
        # Convert the geometry to GeoJSON format
        geojson = mapping(geom)
        
        # Mask the image with the geometry
        masked_img, _ = mask(src, [geojson], crop=True)
        
        # Flatten the image and add it to training data
        training_data.append({
            'X': masked_img.flatten(),
            'y': row['Lam_hyp']  # Assuming 'class_label' is the class column in your field data
        })


# Split the data into training and testing sets
X = [sample['X'] for sample in training_data]
y = [sample['y'] for sample in training_data]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Support Vector Machine (SVM) classifier
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Compute and plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
cm_plot = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, cmap=plt.cm.Blues, display_labels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')
plt.show()


###############################################################################
###############################################################################
###############################################################################
### classify the whole image and plot the result

# Load the PlanetScope tif file for classification
with rasterio.open(PS_imagery_list[0]) as src:
    # Read all bands
    planet_data = src.read()

    # Get image shape
    height, width = planet_data.shape[1:]

    # Reshape the data to have each pixel as a one-dimensional array of eight band values
    flattened_pixels = planet_data.reshape((planet_data.shape[0], -1)).T

flattened_pixels = list(flattened_pixels)


# Perform classification
predicted_classes = clf.predict(flattened_pixels)

# Reshape the predicted classes to the original image shape
predicted_image = predicted_classes.reshape((height, width))

# Plot the classified image
plt.figure(figsize=(10, 8))
plt.imshow(predicted_image, cmap='viridis')  # You can choose a different colormap
plt.title('Classified Image')
plt.colorbar()
plt.show()