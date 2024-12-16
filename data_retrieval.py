import json
import re
import os
import geopandas as gpd
from geopy.geocoders import Nominatim
import pandas as pd
from shapely.geometry import Point, Polygon
import argparse
import yaml
import certifi
import ssl
import geopy.geocoders

import utils
from query_llm import QueryLLM


def initialize_data(data_filename):
    grid_cells_gdf = gpd.read_file('./data/climrr/GridCellsShapefile/GridCells.shp')
    grid_cells_crs = grid_cells_gdf.crs

    if not data_filename.startswith('./data/climrr/'):
        data_filename = f'./data/climrr/{data_filename}'
    try:
        wildfire_df = pd.read_csv(data_filename)
        data_df = wildfire_df
    except FileNotFoundError:
        raise ValueError(f"Invalid data filename {data_filename}.")

    return grid_cells_gdf, grid_cells_crs, data_df


def parse_location(response):
    try:
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            json_string = json_match.group(1)
            parsed_data = json.loads(json_string)
        else:
            return []
    except json.JSONDecodeError:
        print("Invalid JSON format.")
        return []

    if type(parsed_data) is list:
        parsed_data = parsed_data[0]
    if type(parsed_data) is not dict:
        return []

    valid_entries = {}
    for key, item in parsed_data.items():
        if key in ["latitude", "longitude"]:
            valid_entries[key] = item
    if len(valid_entries) < 2:
        return []

    return valid_entries


def get_lat_long(location_description, llm):
    # Initialize Nominatim API
    ctx = ssl._create_unverified_context(cafile=certifi.where())
    geopy.geocoders.options.default_ssl_context = ctx

    geolocator = Nominatim(user_agent="Lauren")

    # Get location
    location = geolocator.geocode(location_description)
    # Extract latitude and longitude
    if location:
        return (location.latitude, location.longitude)
    else:
        messages = f"Output the latitude and longitude of {location_description}. Output a JSON with keys 'latitude' and 'longitude'. Use numbers only. Do not use 'E', 'W'. Use this template: ```json```"
        response = llm.query_llm(self, step='extract_location', content=messages, assistant=False, verbose=False)
        location = parse_location(response)
        if location:
            return (location['latitude'], location['longitude'])
        else:
            return None


def retrieve_crossmodels_within_radius(lat, lon, grid_cells_gdf, grid_cells_crs, geometry='circular', radius=36, verbose=False):
    '''
    Retrieves all Crossmodel indices around a given latitude and longitude,
    using either a circular buffer or a square polygon.

    Parameters:
    - lat: float
        Latitude of the location.
    - lon: float
        Longitude of the location.
    - grid_cells_gdf: GeoDataFrame
        GeoDataFrame of the grid cells.
    - grid_cells_crs: str or dict
        Coordinate Reference System (CRS) of the grid cells (e.g. "EPSG:XXXX").
    - geometry: str, optional
        The geometry shape to use:
        - 'circular': a circular buffer around the point (default)
        - 'square': a square polygon centered on the point
    - radius: float, optional
        The radius size in kilometers if geometry='circular' or the edge size in kilometers if geometry='square'.
    - verbose: bool, optional
        If True, prints the retrieved crossmodel indices.

    Returns:
    - intersecting_cells: GeoDataFrame
        The grid cells intersecting the defined geometry around the point.
    - crossmodel_indices: list
        List containing the Crossmodel indices for the intersecting cells.
    '''

    # Define a default radius in meters for the circular buffer if desired.
    # The original code had a hard-coded 36 km radius. Adjust or parameterize as needed.
    radius_meters = radius * 1000

    # Create the initial point
    point = Point(lon, lat)
    point_gseries = gpd.GeoSeries([point], crs="EPSG:4326")  # Input assumed in WGS84

    # Transform the point to match the grid cells CRS
    point_transformed = point_gseries.to_crs(grid_cells_crs)
    x_center = point_transformed.geometry[0].x
    y_center = point_transformed.geometry[0].y

    if geometry == 'circular':
        # Create a circular buffer
        area_geom = point_transformed.buffer(radius_meters)
        area_geom = area_geom.to_crs(grid_cells_crs)
    elif geometry == 'square':
        # Convert edge size in km to meters
        edge_size_m = radius * 1000

        # Half-edge to define corners
        half_edge = edge_size_m / 2.0

        # Define square polygon in the CRS of the grid
        square_polygon = Polygon([
            (x_center - half_edge, y_center - half_edge),
            (x_center - half_edge, y_center + half_edge),
            (x_center + half_edge, y_center + half_edge),
            (x_center + half_edge, y_center - half_edge)
        ])
        area_geom = gpd.GeoSeries([square_polygon], crs=grid_cells_crs)
    else:
        raise ValueError("Invalid geometry type. Use 'circular' or 'square'.")

    # Find grid cells that intersect the area geometry
    intersecting_cells = grid_cells_gdf[grid_cells_gdf.intersects(area_geom.geometry[0])]

    # Retrieve the Crossmodel indices from the intersecting cells
    crossmodel_indices = intersecting_cells['Crossmodel'].tolist()
    if verbose:
        print('crossmodel_indices', crossmodel_indices)

    return intersecting_cells, crossmodel_indices

# def retrieve_crossmodels_within_radius(lat, lon, grid_cells_gdf, grid_cells_crs, geometry, verbose=False):
#     '''
#     Retrieves all Crossmodel indices within a specified radius of a given latitude and longitude.
#
#     Parameters:
#     - lat: Latitude of the location.
#     - lon: Longitude of the location.
#     - radius_km: The radius in kilometers around the point to retrieve Crossmodel indices.
#     - grid_cells_gdf: GeoDataFrame of the grid cells.
#     - grid_cells_crs: Coordinate Reference System (CRS) of the grid cells.
#
#     Returns:
#     - A list containing the Crossmodel indices for the grid cells within the specified radius.
#     '''
#     # Convert the radius in kilometers to meters (as most CRS use meters)
#     radius_meters = 36 * 1000
#
#     # Create a point from the given latitude and longitude
#     point = Point(lon, lat)
#     point_gseries = gpd.GeoSeries([point], crs="EPSG:4326")  # Assume input is in WGS84
#
#     # Transform the point to match the grid cell CRS
#     point_transformed = point_gseries.to_crs(grid_cells_crs)
#
#     # Create a buffer around the point in the correct CRS
#     buffer = point_transformed.buffer(radius_meters)
#     buffer = buffer.to_crs(grid_cells_crs)
#
#     # Find grid cells that intersect the buffer area
#     intersecting_cells = grid_cells_gdf[grid_cells_gdf.intersects(buffer.geometry[0])]
#
#     # Retrieve the Crossmodel indices from the intersecting cells
#     crossmodel_indices = intersecting_cells['Crossmodel'].tolist()
#     if verbose:
#         print('crossmodel_indices', crossmodel_indices)
#
#     return intersecting_cells, crossmodel_indices


def retrieve_data_from_location(variable, location_description, time_period, llm, geometry, verbose=False):
    """
    This function retrieves the tabular data of the location described in the description within a radius of 36 km.
    """
    # Convert the format of arguments
    data_filename = utils.climate_variables[variable]
    time_period = utils.full_time_frames[variable][time_period]

    # Load the grid cells data
    grid_cells_gdf, grid_cells_crs, data_df = initialize_data(data_filename)

    # Retrieve the latitude and longitude from the response
    location = get_lat_long(location_description, llm)
    if verbose:
        print('location', location)
    lat, lon = location[0], location[1]

    # Retrieve the crossmodel indices in the database
    intersecting_cells, crossmodel_indices = retrieve_crossmodels_within_radius(lat, lon, grid_cells_gdf, grid_cells_crs, geometry, verbose=verbose)

    # Retrieve the data from the database using the crossmodel indices
    data = data_df[data_df['Crossmodel'].isin(intersecting_cells['Crossmodel'])]
    data = data[time_period].values
    if verbose:
        print(data)
    return data


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--city', type=str, default="Chicago", help='To set the city. If you need to enter a space, use the quotes like "Los Angeles, CA"')
    parser.add_argument('--time', type=str, default="'spring in historical period'", help='To set the time period. Check all available time periods in climate_variables in utils.py')
    parser.add_argument('--var', type=str, default="'fire weather index'", help='To set the climate variable. Check all available variables in full_time_frames in utils.py')
    parser.add_argument('--geometry', type=str, default="square", help='To set the geometry of the location. Choose from "square", "circular", or "real"')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')

    cmd_args = parser.parse_args()
    cmd_args.city = cmd_args.city.replace("'", '')
    cmd_args.time = cmd_args.time.replace("'", '')
    cmd_args.var = cmd_args.var.replace("'", '')

    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')
    llm = QueryLLM(args)

    retrieve_data_from_location(cmd_args.var, cmd_args.city, cmd_args.time, llm, cmd_args.geometry, cmd_args.verbose)
