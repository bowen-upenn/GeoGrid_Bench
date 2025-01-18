import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os
from time import sleep
import requests
import json
import ast

import folium
from folium.raster_layers import ImageOverlay
import geopandas as gpd
from shapely.geometry import Point, Polygon

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

import ocr


def categorize_fwi(value):
    """Categorize the FWI value into its corresponding class and return the value and category."""
    if value <= 9:
        return 'Low'
    elif value <= 21:
        return 'Medium'
    elif value <= 34:
        return 'High'
    elif value <= 39:
        return 'Very High'
    elif value <= 53:
        return 'Extreme'
    else:
        return 'Very Extreme'


def fwi_color(value):
    fwi_class_colors = {
        'Low': 'rgb(255, 255, 0, 0.5)',
        'Medium': 'rgb(255, 204, 0, 0.5)',
        'High': 'rgb(255, 153, 0, 0.5)',
        'Very High': 'rgb(255, 102, 0, 0.5)',
        'Extreme': 'rgb(255, 51, 0, 0.5)',
        'Very Extreme': 'rgb(255, 0, 0, 0.5)'
    }
    return fwi_class_colors[categorize_fwi(value)]


def fwi_color_plt(value):
    fwi_class_colors = {
        'Low': (1.0, 1.0, 0.0, 0.5),  # Yellow with 50% transparency
        'Medium': (1.0, 0.8, 0.0, 0.5),  # Orange with 50% transparency
        'High': (1.0, 0.6, 0.0, 0.5),  # Darker orange with 50% transparency
        'Very High': (1.0, 0.4, 0.0, 0.5),  # Even darker orange with 50% transparency
        'Extreme': (1.0, 0.2, 0.0, 0.5),  # Red with 50% transparency
        'Very Extreme': (1.0, 0.0, 0.0, 0.5)  # Dark red with 50% transparency
    }
    return fwi_class_colors[categorize_fwi(value)]


def add_legend():
        """Adds a custom legend with colors for FWI to a Folium map."""
        legend_html = """
        <div style="
                    bottom: 5px; left: 5px; width: auto; height: 50px; 
                    border:1px solid grey; z-index:9999; font-size:12px;
                    background: white; opacity: 0.9; padding: 2px; color: black; display: flex; align-items: center;justify-content: center;">
        <i style="background:rgb(255, 255, 0, 0.5); width: 24px; height: 24px;"></i> &nbsp; Low &nbsp;|&nbsp;
        <i style="background:rgb(255, 204, 0, 0.5); width: 24px; height: 24px;"></i> &nbsp; Medium &nbsp;|&nbsp;
        <i style="background:rgb(255, 153, 0, 0.5); width: 24px; height: 24px;"></i> &nbsp; High &nbsp;|&nbsp;
        <i style="background:rgb(255, 102, 0, 0.5); width: 24px; height: 24px;"></i> &nbsp; Very High &nbsp;|&nbsp;
        <i style="background:rgb(255, 51, 0, 0.5); width: 24px; height: 24px;"></i> &nbsp; Extreme &nbsp;|&nbsp;
        <i style="background:rgb(255, 0, 0, 0.5); width: 24px; height: 24px;"></i> &nbsp; Very Extreme
        </div>
        """
        # write in the middle of the page
        st.write(legend_html, unsafe_allow_html=True)


def get_map(crossmodels, df, period, season='spring'):
    season_columns = [col for col in df.columns if season in col and period in col]
    season_fwi_df = df[['Crossmodel'] + season_columns]
    col_name = season_columns[0]
    season_fwi_df.loc[:, 'class'] = season_fwi_df[col_name].apply(categorize_fwi)

    fwi_df_geo = gpd.GeoDataFrame(crossmodels.merge(season_fwi_df, left_on='Crossmodel', right_on='Crossmodel'))

    m = folium.Map(location=st.session_state.center, zoom_start=st.session_state.zoom)
    m.add_child(
        folium.features.GeoJson(fwi_df_geo,
            tooltip=folium.features.GeoJsonTooltip(fields=['Crossmodel', col_name, 'class'], aliases=['Crossmodel', 'FWI', 'class']),
            style_function=lambda x: {'fillColor': fwi_color(x['properties'][col_name]),
                                      'color': fwi_color(x['properties'][col_name])})
    )
    return m


def overlay_heatmap_on_map(data_df, matrix, variable_name, time_period, cell_geometries, color_norm, center_lat, center_lon, size_km=64, alpha=True, output_path="heatmap_map.png", verbose=False):
    # Extract required columns
    variable_columns = [col for col in data_df.columns if time_period == col]
    variable_df = data_df[['Crossmodel'] + variable_columns]
    col_name = variable_columns[0]
    variable_df['class'] = variable_df[col_name].apply(categorize_fwi)

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
    print("Columns after cleanup:", data_df_geo.columns)

    # Check if required fields exist
    required_fields = ['Crossmodel', col_name, 'class', 'geometry']
    for field in required_fields:
        if field not in data_df_geo.columns:
            raise ValueError(f"Missing required field '{field}' in data_df_geo.")

    # Create the map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9)
    m.add_child(
        folium.features.GeoJson(
            data_df_geo,
            tooltip=folium.features.GeoJsonTooltip(fields=['Crossmodel', col_name, 'class']),
        )
    )

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
