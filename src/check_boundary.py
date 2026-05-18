import geopandas as gpd

gdf = gpd.read_file("data_raw/boundaries/geoBoundaries-PAK-ADM1.shp")
sindh = gdf[gdf["shapeName"] == "Sindh"]
sindh.to_file("data_raw/boundaries/sindh_boundary.shp")
print("Saved Sindh boundary.")
