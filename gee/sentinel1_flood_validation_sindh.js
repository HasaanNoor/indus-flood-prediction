// Sentinel-1 Flood Validation for Sindh, Pakistan
// Pilot event: 2019-07-02 to 2019-08-24

var eventStart = '2019-07-02';
var eventEnd = '2019-08-24';

// Baseline before flood
var beforeStart = '2019-05-15';
var beforeEnd = '2019-06-25';

// Approx Sindh bounding box.
// Later we can replace this with exact Sindh boundary GeoJSON.
var sindh = ee.Geometry.Rectangle([66.5, 23.5, 71.2, 28.6]);

Map.centerObject(sindh, 7);
Map.addLayer(sindh, {color: 'red'}, 'Sindh AOI');

// Sentinel-1 collection
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(sindh)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('resolution_meters', 10))
  .select('VH');

// Pre-flood and during-flood composites
var before = s1
  .filterDate(beforeStart, beforeEnd)
  .median()
  .clip(sindh);

var during = s1
  .filterDate(eventStart, eventEnd)
  .median()
  .clip(sindh);

// Flooded water often appears darker in VH.
// So we look for pixels where during-flood VH is lower than before-flood VH.
var difference = during.subtract(before);

// Previous threshold was -1.5 and was too conservative.
// Updated threshold: -0.8
var flood = difference.lt(-0.8);

// Mask permanent water using JRC Global Surface Water
var gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater');
var permanentWater = gsw.select('seasonality').gte(10);
flood = flood.where(permanentWater, 0);

// Mask steep slopes using SRTM
var dem = ee.Image('USGS/SRTMGL1_003');
var slope = ee.Terrain.slope(dem);
flood = flood.where(slope.gt(5), 0);

// Remove tiny noisy patches
var connectedPixels = flood.connectedPixelCount(25);
flood = flood.updateMask(connectedPixels.gte(8));

// Calculate detected flood area in km2
var floodArea = flood
  .multiply(ee.Image.pixelArea())
  .divide(1e6)
  .reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: sindh,
    scale: 30,
    maxPixels: 1e13
  });

// Print diagnostics
print('Sentinel-1 images before flood:', s1.filterDate(beforeStart, beforeEnd).size());
print('Sentinel-1 images during flood:', s1.filterDate(eventStart, eventEnd).size());
print('Flood area km2:', floodArea);

// Visualization layers
Map.addLayer(before, {min: -25, max: 0}, 'Before flood VH');
Map.addLayer(during, {min: -25, max: 0}, 'During flood VH');
Map.addLayer(
  difference,
  {min: -5, max: 5, palette: ['blue', 'white', 'red']},
  'VH difference'
);
Map.addLayer(flood, {palette: ['0000ff']}, 'Detected flood extent');

// Export flood mask to Google Drive
Export.image.toDrive({
  image: flood,
  description: 'sentinel1_flood_mask_sindh_2019_event1_threshold_08',
  folder: 'indus_flood_validation',
  fileNamePrefix: 'sentinel1_flood_mask_sindh_2019_event1_threshold_08',
  region: sindh,
  scale: 30,
  maxPixels: 1e13
});