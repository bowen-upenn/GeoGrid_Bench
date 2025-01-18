import numpy as np
import pandas as pd
import io
import base64
import os
from time import sleep
import requests
import json
import ast

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from PIL import Image, ImageDraw, ImageFont
import folium
from folium.raster_layers import ImageOverlay
from folium import Html, Element
import geopandas as gpd
import branca.colormap as cm

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

import ocr


def overlay_heatmap_on_map(data_df, matrix, variable_name, time_period, cell_geometries, color_norm, center_lat, center_lon, size_km=64, alpha=True, output_path="heatmap_map.png", verbose=False):
    # Extract required columns
    variable_columns = [col for col in data_df.columns if time_period == col]
    variable_df = data_df[['Crossmodel'] + variable_columns]
    col_name = variable_columns[0]

    # Merge with geometries
    variable_df = variable_df.merge(cell_geometries, on='Crossmodel', how='inner')
    data_df_geo = gpd.GeoDataFrame(
        data_df.merge(variable_df, on='Crossmodel', how='inner', suffixes=('_x', '_y'))
    )

    # Remove any other duplicated columns
    if 'hist_x' in data_df_geo.columns and 'hist_y' in data_df_geo.columns:
        data_df_geo['hist'] = data_df_geo['hist_x']  # Or choose 'hist_y'
        data_df_geo.drop(columns=['hist_x', 'hist_y'], inplace=True)
    data_df_geo = data_df_geo.loc[:, ~data_df_geo.columns.duplicated()]

    # Check if required fields exist
    required_fields = ['Crossmodel', col_name, 'geometry']
    for field in required_fields:
        if field not in data_df_geo.columns:
            raise ValueError(f"Missing required field '{field}' in data_df_geo.")

    # Function to map numerical value to color in the provided color_norm
    def value_to_color(value):
        if value is None or np.isnan(value):
            return 'rgba(0,0,0,0)'
        color = colormap(value)
        return color

    # colormap = cm.linear.RdYlBu_10.scale(color_norm.vmin, color_norm.vmax)
    colormap = cm.LinearColormap(
        colors=['#276AAE', '#7FB6D5', '#E6E1DE', '#F39F7B', '#B41D2E'],
        vmin=color_norm.vmin,
        vmax=color_norm.vmax
    )
    data_df_geo['color'] = data_df_geo[col_name].apply(value_to_color)

    # Create the map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9)
    m.add_child(
        folium.features.GeoJson(
            data_df_geo,
            style_function=lambda feature: {
                'fillColor': feature['properties']['color'],
                'color': 'black',  # border color
                'weight': 0.5,
                'fillOpacity': 0.5,
            },
            tooltip=folium.features.GeoJsonTooltip(fields=['Crossmodel', col_name]),
        )
    )
    colormap.caption = "Values"
    m.add_child(colormap)

    # Save map as HTML
    m.save(f"temp_map{output_path[-5]}.html")

    # Use Selenium to save as image
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(service=ChromeService(), options=chrome_options)

    # Set window size to match map dimensions based on block size and km-to-pixel ratio
    rows, cols = matrix.shape
    km_per_block = 12
    km_to_pixel_ratio = 8

    window_width = int(cols * km_per_block * km_to_pixel_ratio)
    window_height = int(rows * km_per_block * km_to_pixel_ratio)
    driver.set_window_size(window_width, window_height)

    driver.get("file://" + os.path.abspath(f"temp_map{output_path[-5]}.html"))
    sleep(3)  # Wait for map to load
    driver.save_screenshot(output_path)
    screenshot = Image.open(output_path)
    width, height = screenshot.size
    if verbose:
        print(f"Map saved as image: {output_path} with size {width}x{height} pixels")
    driver.quit()

    return screenshot, width, height


def visualize_heatmap(matrix, variable_name, color_norm, output_path="heatmap", verbose=False):
    # Flip the matrix upside down to match the map orientation
    matrix = matrix.iloc[::-1]

    # Normalize the matrix for color mapping
    norm = Normalize(vmin=matrix.min().min(), vmax=matrix.max().max())
    colormap = plt.get_cmap('coolwarm')

    # Create the heatmap visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = ax.imshow(matrix, cmap=colormap, norm=color_norm)

    # Add column and row indices as red bold text
    for i, row in enumerate(matrix.index):
        for j, col in enumerate(matrix.columns):
            value = matrix.iloc[i, j]
            if not np.isnan(value):
                ax.text(j, i, f'{value:.1f}', ha='center', va='center', fontsize=8, color='black')
            else:
                ax.text(j, i, 'N/A', ha='center', va='center', fontsize=8, color='black')
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels([f'C{col}' for col in matrix.columns], fontsize=10, color='red', fontweight='bold')
    ax.set_yticklabels([f'R{row}' for row in matrix.index], fontsize=10, color='red', fontweight='bold')

    # Add color bar
    cbar = plt.colorbar(heatmap, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label('Values', fontsize=12)

    # Save and display the heatmap with indices
    plt.title(variable_name.capitalize() + ' heatmap with row and column indices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()

    heatmap = Image.open(output_path)
    if verbose:
        print(f"Heatmap saved as: {output_path}")
    return heatmap, colormap, norm



def visualize_grids(data_df, matrix, variable_name, time_period, cell_geometries, color_norm, center_lat, center_lon, size_km=64, output_path="heatmap", verbose=False):
    """
    Overlay a heatmap from a matrix onto a real map centered at the given latitude and longitude, and save as an image.
    """
    # Normalize the matrix for color mapping
    heatmap, colormap, norm = visualize_heatmap(matrix, variable_name, color_norm, output_path=f"{output_path}.png", verbose=verbose)

    # Draw the final image with transparency on maps
    overlay_path = f"{output_path[:-1]}_overlay{output_path[-1]}.png"
    overlay, overlay_width, overlay_height = overlay_heatmap_on_map(data_df, matrix, variable_name, time_period, cell_geometries, color_norm, center_lat, center_lon, size_km, alpha=True, output_path=overlay_path, verbose=verbose)

    return heatmap, overlay, overlay_path, overlay_width, overlay_height
