// Phase 14 Sentinel-1 event configuration for Sindh multi-event labels.
// Keep this file synchronized with:
// data_processed/spatial/labels/inventory/sentinel_event_inventory.json

var SENTINEL_EVENTS = [
  {
    event_id: '2015_sindh_monsoon_candidate',
    event_name: '2015 Sindh monsoon flood candidate',
    event_start: '2015-07-20',
    event_end: '2015-08-20',
    baseline_start: '2015-06-10',
    baseline_end: '2015-07-10',
    polarization: 'VH',
    orbit_pass: 'DESCENDING',
    threshold: -3.0,
    threshold_alternatives: [-2.4, -3.0, -3.5],
    permanent_water_occurrence_threshold: 90,
    slope_threshold_degrees: 5,
    connected_pixel_threshold: 8,
    export_enabled: false
  },
  {
    event_id: '2019_sindh_monsoon_event_01',
    event_name: '2019 Sindh monsoon flood validation event',
    event_start: '2019-08-01',
    event_end: '2019-08-15',
    baseline_start: '2019-06-15',
    baseline_end: '2019-06-30',
    polarization: 'VH',
    orbit_pass: 'DESCENDING',
    threshold: -3.0,
    threshold_alternatives: [-2.4, -3.0],
    permanent_water_occurrence_threshold: 90,
    slope_threshold_degrees: 5,
    connected_pixel_threshold: 8,
    export_enabled: true
  },
  {
    event_id: '2020_sindh_monsoon_candidate',
    event_name: '2020 Sindh monsoon flood candidate',
    event_start: '2020-08-20',
    event_end: '2020-09-20',
    baseline_start: '2020-07-01',
    baseline_end: '2020-07-31',
    polarization: 'VH',
    orbit_pass: 'DESCENDING',
    threshold: -3.0,
    threshold_alternatives: [-2.4, -3.0, -3.5],
    permanent_water_occurrence_threshold: 90,
    slope_threshold_degrees: 5,
    connected_pixel_threshold: 8,
    export_enabled: false
  },
  {
    event_id: '2022_pakistan_sindh_event_01',
    event_name: '2022 Pakistan flood affecting Sindh',
    event_start: '2022-08-15',
    event_end: '2022-09-20',
    baseline_start: '2022-06-15',
    baseline_end: '2022-07-15',
    polarization: 'VH',
    orbit_pass: 'DESCENDING',
    threshold: -3.0,
    threshold_alternatives: [-2.4, -3.0, -3.5],
    permanent_water_occurrence_threshold: 90,
    slope_threshold_degrees: 5,
    connected_pixel_threshold: 8,
    export_enabled: false
  }
];
