// Sentinel-1 Flood Validation for Sindh, Pakistan
// Pilot flood event refinement

// Peak flood period
var eventStart = '2019-08-01';
var eventEnd = '2019-08-15';

// Baseline period
var beforeStart = '2019-06-15';
var beforeEnd = '2019-06-30';

// Approximate Sindh boundary
var sindh = ee.Geometry.Rectangle([66.5, 23.5, 71.2, 28.6]);

Map.centerObject(sindh, 7);
Map.addLayer(sindh, {color: 'red'}, 'Sindh AOI');

// Sentinel-1 VH imagery
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(sindh)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('resolution_meters', 10))
  .select('VH');

// Pre-flood composite
var before = s1
  .filterDate(beforeStart, beforeEnd)
  .median()
  .clip(sindh);

// During-flood composite
var during = s1
  .filterDate(eventStart, eventEnd)
  .median()
  .clip(sindh);

// Difference image
var difference = during.subtract(before);

// Examine the distribution of VH changes
var stats = difference.reduceRegion({
  reducer: ee.Reducer.percentile([1, 5, 10, 25, 50, 75, 90, 95, 99]),
  geometry: sindh,
  scale: 100,
  maxPixels: 1e13
});

print('Difference statistics', stats);

// Diagnostics
print(
  'Sentinel-1 images before flood:',
  s1.filterDate(beforeStart, beforeEnd).size()
);

print(
  'Sentinel-1 images during flood:',
  s1.filterDate(eventStart, eventEnd).size()
);

// Threshold based on observed VH difference distribution.
// p10 was around -2.37, so -2.4 captures strongest negative-change pixels.
var floodRaw = difference.lt(-3.0);

// Mask permanent water using JRC Global Surface Water
var gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater');
var permanentWater = gsw.select('seasonality').gte(10);
var flood = floodRaw.where(permanentWater, 0);

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

print('Cleaned flood area km2:', floodArea);

// Visualization
Map.addLayer(
  before,
  {min: -25, max: 0},
  'Before flood VH'
);

Map.addLayer(
  during,
  {min: -25, max: 0},
  'During flood VH'
);

Map.addLayer(
  difference,
  {
    min: -3,
    max: 3,
    palette: ['blue', 'white', 'red']
  },
  'VH difference'
);

Map.addLayer(
  gsw.select('occurrence'),
  {
    min: 0,
    max: 100
  },
  'JRC Water Occurrence',
  false
);

Map.addLayer(
  floodRaw.selfMask(),
  {palette: ['cyan']},
  'Raw flood threshold -2.4',
  false
);

Map.addLayer(
  flood.selfMask(),
  {palette: ['0000ff']},
  'Cleaned flood extent -2.4'
);

// Export cleaned flood mask
Export.image.toDrive({
  image: flood.selfMask(),
  description: 'sentinel1_flood_mask_sindh_2019_event1_threshold_24',
  folder: 'indus_flood_validation',
  fileNamePrefix: 'sentinel1_flood_mask_sindh_2019_event1_threshold_24',
  region: sindh,
  scale: 30,
  maxPixels: 1e13
});