// define asset ID of uploaded .tif file (bathymetry raster which defines the AOI was uploaded to GEE beforehand)
var tifAsset = 'users/exampljo/24-04-29_Bathy_MLWS_5m_raster_aoi_clipped';

// load bathymetry raster and get its extent
var image = ee.Image(tifAsset);
var bounds = image.geometry();

// define time range for data acquisition (April to August 2023)
var startDate = '2023-04-01';
var endDate = '2023-08-31';

// define a simple cloud masking function for each S2 image using the QA band
function maskS2Clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000); // rescale the image on the fly
}

// load S2 reflectance data and apply cloud masking function
var sentinel2SR = ee.ImageCollection('COPERNICUS/S2_SR')
    .filterDate(startDate, endDate)
    .filterBounds(bounds)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(maskS2Clouds);

// define a function to resample image to 5 meters using bilinear interpolation
function resampleImage(image) {
  return image.resample('bilinear').reproject({
    crs: 'EPSG:32632', // Adjust CRS if needed
    scale: 5
  });
}

// apply resampling function to each image before calculating indices
var sentinel2Resampled = sentinel2SR.map(resampleImage);

// define a function to calculate to select bands and calculate indices of interest
function addIndices(image) {
  var B2 = image.select('B2'); // blue band
  var B3 = image.select('B3'); // green band
  var B4 = image.select('B4'); // red band
  var B8 = image.select('B8'); // NIR band
  var ndti = B4.subtract(B3).divide(B4.add(B3)).rename('NDTI');
  var ndwi = B3.subtract(B8).divide(B3.add(B8)).rename('NDWI');
  var ndavi = B8.subtract(B2).divide(B8.add(B2)).rename('NDAVI');
  var wavi = B8.subtract(B2).multiply(1 + 0.5).divide(B8.add(B2).add(0.5)).rename('WAVI');
  var grvi = B3.subtract(B4).divide(B3.add(B4)).rename('GRVI');
  var npci = B4.subtract(B2).divide(B4.add(B2)).rename('NPCI');
  var bgRatio = B2.divide(B3).rename('BG_RATIO'); // simple B/G ratio
  var ndvi = B8.subtract(B4).divide(B8.add(B4)).rename('NDVI'); 

  return image.addBands([ndti, ndwi, ndavi, wavi, grvi, npci, bgRatio, ndvi]);
}

// apply the index calculations to each image
var sentinel2WithIndices = sentinel2Resampled.map(addIndices);

// reduce the image collection to a single image which holds median for each index and band and crop it to AOI
var indices = ['NDTI', 'NDWI', 'NDAVI', 'WAVI', 'GRVI', 'NPCI', 'BG_RATIO', 'NDVI'];
var bands = ['B2', 'B3', 'B4', 'B8'];
var medianImages = {};
indices.forEach(function(index) {
  medianImages[index] = sentinel2WithIndices.select(index).median().clip(bounds);
});
bands.forEach(function(band) {
  medianImages[band] = sentinel2WithIndices.select(band).median().clip(bounds);
});

// visualize the results in GEE
Map.centerObject(bounds, 10);
Object.keys(medianImages).forEach(function(index) {
  Map.addLayer(medianImages[index], {min: -1, max: 1, palette: ['blue', 'white', 'green']}, index + ' Median');
});
Map.addLayer(bounds, {}, 'AOI');

// export the derived bands and indices to Google Drive and set CRS
indices.concat(bands).forEach(function(index) {
  Export.image.toDrive({
    image: medianImages[index],
    description: index + 'Median',
    folder: 'EarthEngineExports',
    fileNamePrefix: index.toLowerCase() + '_median_2023',
    region: bounds,
    scale: 5,  
    crs: 'EPSG:32632'
  });
});
