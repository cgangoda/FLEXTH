# FLEXTH - Flood extent enhancement and water depth estimation tool for satellite-derived inundation maps

FLEXTH is a tool to enhance flood maps (e.g. satellite-derived) by accounting for terrain topography. It expands inundations to areas which are likely to be flooded based on their altimetry and provides estimates of water levels and water depths. The algorithm requires, as a primary input, a flood delineation map and a DTM. Additional information may include areas excluded from flood mapping and/or permanent water bodies. All input must be provided via georeferenced binary rasters (GeoTIFF) in a suitable projected reference system.

Run FLEXTH using the script "FLEXTH.py" contained in this repository.

Alternatively, the script "FLEXTH_tiling.py" automatically tiles your inputs and runs FLEXTH in sequence across the tiles. This can be useful for very large areas and/or high resolutions and/or when computational resources (especially memory) are limited. 

An additional script is provided named "DTM_2_floodmap.py" which can easily help you resample and reproject your input DTM (or any other input, e.g. a permanent water body mask) into the same grid, extent and reference system as your flood map raster. 

See the script files for further details. 

Since water depth is the primary proxy for flood damages, the tool facilitates flood impact assessment over large scales with minimum supervision and quick computational times.



CITE :  A.Betterle & P.Salamon - Water depth estimate and flood extent enhancement for satellite-based inundation maps (2024) - NHESS - https://doi.org/10.5194/nhess-24-2817-2024



For further information contact: 

andrea.betterle@ec.europa.eu  
peter.salamon@ec.europa.eu
