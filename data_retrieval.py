import json
import re
import os
import geopandas as gpd
from geopy.geocoders import Nominatim
import pandas as pd
from shapely.geometry import Point
import argparse
import yaml

import certifi
import ssl
import geopy.geocoders
from geopy.geocoders import Nominatim

from query_llm import QueryLLM


def initialize_data(data_filename):
    grid_cells_gdf = gpd.read_file('./data/climrr/GridCellsShapefile/GridCells.shp')
    grid_cells_crs = grid_cells_gdf.crs

    if data_filename == 'FireWeatherIndex_Wildfire.csv':
        wildfire_df = pd.read_csv('./data/climrr/FireWeatherIndex_Wildfire.csv')
        data_df = wildfire_df
    else:
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


def retrieve_crossmodels_within_radius(lat, lon, grid_cells_gdf, grid_cells_crs):
    '''
    Retrieves all Crossmodel indices within a specified radius of a given latitude and longitude.

    Parameters:
    - lat: Latitude of the location.
    - lon: Longitude of the location.
    - radius_km: The radius in kilometers around the point to retrieve Crossmodel indices.
    - grid_cells_gdf: GeoDataFrame of the grid cells.
    - grid_cells_crs: Coordinate Reference System (CRS) of the grid cells.

    Returns:
    - A list containing the Crossmodel indices for the grid cells within the specified radius.
    '''
    # Convert the radius in kilometers to meters (as most CRS use meters)
    radius_meters = 36 * 1000

    # Create a point from the given latitude and longitude
    point = Point(lon, lat)
    point_gseries = gpd.GeoSeries([point], crs="EPSG:4326")  # Assume input is in WGS84

    # Transform the point to match the grid cell CRS
    point_transformed = point_gseries.to_crs(grid_cells_crs)

    # Create a buffer around the point in the correct CRS
    buffer = point_transformed.buffer(radius_meters)
    buffer = buffer.to_crs(grid_cells_crs)

    # Find grid cells that intersect the buffer area
    intersecting_cells = grid_cells_gdf[grid_cells_gdf.intersects(buffer.geometry[0])]

    # Retrieve the Crossmodel indices from the intersecting cells
    crossmodel_indices = intersecting_cells['Crossmodel'].tolist()
    print('crossmodel_indices', crossmodel_indices)

    return intersecting_cells, crossmodel_indices


def retrieve_data_from_location(data_filename, location_description, time_period, llm):
    """
    This function retrieves the tabular data of the location described in the description within a radius of 36 km.
    """
    # Load the grid cells data
    grid_cells_gdf, grid_cells_crs, data_df = initialize_data(data_filename)

    # Retrieve the latitude and longitude from the response
    location = get_lat_long(location_description, llm)
    print('location', location)
    lat, lon = location[0], location[1]

    # Retrieve the crossmodel indices in the database
    intersecting_cells, crossmodel_indices = retrieve_crossmodels_within_radius(lat, lon, grid_cells_gdf, grid_cells_crs)

    print('intersecting_cells', intersecting_cells['geometry'].iloc[0])

    # Retrieve the data from the database using the crossmodel indices
    data = data_df[data_df['Crossmodel'].isin(intersecting_cells['Crossmodel'])]
    data = data[time_period]
    print(data)
    return data


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--city', type=str, default="chicago", help='To set the city. If you need to enter a space, use the quotes like "Los Angeles, CA"')
    parser.add_argument('--time', type=str, default="wildfire_autumn_Endc", help='To set the time period')
    parser.add_argument('--dataset', type=str, default="FireWeatherIndex_Wildfire", help='To set the dataset name')
    cmd_args = parser.parse_args()
    cmd_args.dataset += '.csv'

    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')
    llm = QueryLLM(args)

    retrieve_data_from_location(cmd_args.dataset, cmd_args.city, cmd_args.time, llm)
