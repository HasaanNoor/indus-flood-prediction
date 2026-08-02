// Phase 14 Sentinel-1 multi-event flood-label workflow for Sindh, Pakistan.
//
// In the Earth Engine Code Editor, paste sentinel1_event_config.js above this
// script or import it as a required script so SENTINEL_EVENTS is defined.

var sindh = ee.Geometry.Rectangle([66.5, 23.5, 71.2, 28.6]);
var exportFolder = 'indus_flood_validation';
var selectedEventIds = []; // Empty means process all configured events. Example: ['2022_pakistan_sindh_event_01']
var printThresholdDiagnostics = true;

Map.centerObject(sindh, 7);
Map.addLayer(sindh, {color: 'red'}, 'Sindh AOI');

function getS1Collection(eventConfig) {
  var collection = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(sindh)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', eventConfig.polarization))
    .select(eventConfig.polarization);
  if (eventConfig.orbit_pass !== null && eventConfig.orbit_pass !== undefined && eventConfig.orbit_pass !== 'ANY') {
    collection = collection.filter(ee.Filter.eq('orbitProperties_pass', eventConfig.orbit_pass));
  }
  return collection;
}

function areaKm2(mask) {
  return mask
    .selfMask()
    .multiply(ee.Image.pixelArea())
    .divide(1e6)
    .reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: sindh,
      scale: 30,
      maxPixels: 1e13
    });
}

function buildPermanentWater(eventConfig) {
  var gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater');
  return gsw.select('occurrence').gte(eventConfig.permanent_water_occurrence_threshold).rename('permanent_water');
}

function cleanFloodMask(rawFlood, eventConfig) {
  var permanentWater = buildPermanentWater(eventConfig);
  var dem = ee.Image('USGS/SRTMGL1_003');
  var slope = ee.Terrain.slope(dem).rename('slope_degrees');
  var candidate = rawFlood
    .where(permanentWater, 0)
    .where(slope.gt(eventConfig.slope_threshold_degrees), 0)
    .rename('candidate_observed_inundation');
  var connectedPixels = candidate.connectedPixelCount(25, true);
  var cleaned = candidate.updateMask(connectedPixels.gte(eventConfig.connected_pixel_threshold));
  return cleaned.unmask(0).uint8().rename('candidate_observed_inundation');
}

function processSentinelEvent(eventConfig) {
  var s1 = getS1Collection(eventConfig);
  var beforeCollection = s1.filterDate(eventConfig.baseline_start, eventConfig.baseline_end);
  var duringCollection = s1.filterDate(eventConfig.event_start, eventConfig.event_end);
  var before = beforeCollection.median().clip(sindh);
  var during = duringCollection.median().clip(sindh);
  var difference = during.subtract(before).rename('difference_db');
  var percentiles = difference.reduceRegion({
    reducer: ee.Reducer.percentile([1, 5, 10, 25, 50, 75, 90, 95, 99]),
    geometry: sindh,
    scale: 100,
    maxPixels: 1e13
  });

  print('Event', eventConfig.event_id, eventConfig.event_name);
  print('Expected local source_mask_path', 'outputs/validation/sentinel1/sentinel1_flood_mask_sindh_' + eventConfig.event_id + '.tif');
  print('Baseline window', eventConfig.baseline_start, eventConfig.baseline_end);
  print('Event window', eventConfig.event_start, eventConfig.event_end);
  print('Polarization/orbit', eventConfig.polarization, eventConfig.orbit_pass);
  print('Baseline image count', beforeCollection.size());
  print('During image count', duringCollection.size());
  print('Difference percentiles', percentiles);

  var rawFlood = difference.lt(eventConfig.threshold).rename('raw_threshold_flood');
  var cleaned = cleanFloodMask(rawFlood, eventConfig);
  var permanentWater = buildPermanentWater(eventConfig).uint8();
  print('Selected threshold', eventConfig.threshold);
  print('Cleaned flood area km2', areaKm2(cleaned));
  print('Permanent water area km2', areaKm2(permanentWater));

  if (printThresholdDiagnostics) {
    eventConfig.threshold_alternatives.forEach(function(threshold) {
      var sensitivity = cleanFloodMask(difference.lt(threshold), eventConfig);
      print('Threshold sensitivity area km2 ' + eventConfig.event_id + ' ' + threshold, areaKm2(sensitivity));
    });
  }
  var exportImage = cleaned.addBands(permanentWater).clip(sindh);

  Map.addLayer(before, {min: -25, max: 0}, eventConfig.event_id + ' before ' + eventConfig.polarization, false);
  Map.addLayer(during, {min: -25, max: 0}, eventConfig.event_id + ' during ' + eventConfig.polarization, false);
  Map.addLayer(difference, {min: -4, max: 4, palette: ['blue', 'white', 'red']}, eventConfig.event_id + ' difference', false);
  Map.addLayer(cleaned.selfMask(), {palette: ['0000ff']}, eventConfig.event_id + ' candidate observed inundation', eventConfig.export_enabled);
  Map.addLayer(permanentWater.selfMask(), {palette: ['00ffff']}, eventConfig.event_id + ' permanent water mask', false);

  if (eventConfig.export_enabled) {
    Export.image.toDrive({
      image: exportImage,
      description: 'sentinel1_flood_mask_sindh_' + eventConfig.event_id,
      folder: exportFolder,
      fileNamePrefix: 'sentinel1_flood_mask_sindh_' + eventConfig.event_id,
      region: sindh,
      scale: 30,
      maxPixels: 1e13
    });
  }
}

function shouldProcess(eventConfig) {
  return selectedEventIds.length === 0 || selectedEventIds.indexOf(eventConfig.event_id) !== -1;
}

SENTINEL_EVENTS.filter(shouldProcess).forEach(processSentinelEvent);
