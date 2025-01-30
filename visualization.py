import numpy as np
import pandas as pd
import io
import base64
import os
from time import sleep
import requests
import json
import ast
import math
import cv2

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from PIL import Image, ImageDraw, ImageFont
import folium
from folium.raster_layers import ImageOverlay
from folium import Html, Element
import geopandas as gpd
from shapely.geometry import Point
import branca.colormap as cm
from pyproj import Transformer

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

import ocr


def find_grid_angle(image):
    image = np.array(image)
    distance_to_black = np.linalg.norm(image - np.array([0, 0, 0]), axis=-1)
    grid_mask = distance_to_black <= 80
    filtered_image = np.zeros_like(image)
    filtered_image[grid_mask] = np.array([255, 255, 255])
    cv2.imwrite('filtered_image.jpg', filtered_image)

    grey = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(grey, 50, 150, apertureSize=7)
    kernel = np.ones((5, 5), np.uint8)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
    lines = cv2.HoughLines(edge, 1, np.pi / 180, 200)  # Adjust the threshold value (200) as needed
    cv2.imwrite('edge.jpg', edge)

    angles = []
    horizontal_lines = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta)  # Convert theta to degrees

            # Filter for horizontal lines (angles near 0° or 180°)
            if abs(angle) < 45 or abs(angle - 180) < 45:
                horizontal_lines.append((rho, theta))
                angles.append(angle)

    angle = np.median(angles)
    if angle < 90:
        angle = -angle
    else:
        angle = 180 - angle
    print("Detected line angle:", angle)

    return angle


def add_crossmodel_indices_on_map(data_df_geo, m):
    # Helper functions to extract numeric parts from row and column labels
    def extract_row_id(row_label):
        return int(row_label[1:])  # Extract number after 'R'

    def extract_col_id(col_label):
        return int(col_label[1:])  # Extract number after 'C'

    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    all_indices = []
    for index, row in data_df_geo.iterrows():
        all_indices.append((row['Crossmodel'][:4], row['Crossmodel'][4:]))

    # Initialize dictionaries to store results
    smallest_column_per_row = {}
    largest_row_per_column = {}

    # Process all indices
    for row, col in all_indices:
        # Update smallest column for each row
        if row not in smallest_column_per_row or extract_col_id(col) < extract_col_id(smallest_column_per_row[row]):
            smallest_column_per_row[row] = col

        # Update largest row for each column
        if col not in largest_row_per_column or extract_row_id(row) > extract_row_id(largest_row_per_column[col]):
            largest_row_per_column[col] = row

    # grid shapes may vary based on the map, especially along ocean coasts or national boarders, so we calculate the average grid size
    grid_width, grid_height = [], []
    count_grid = 0
    for index, row in data_df_geo.iterrows():  # Assuming data_df_geo contains the geometries
        geometry = row['geometry']
        if geometry.is_empty:
            continue
        try:
            # find the width of each grid cell
            all_lon = []
            all_lat = []
            for point in geometry.exterior.coords:
                lon, lat = transformer.transform(point[0], point[1])
                all_lon.append(lon)
                all_lat.append(lat)
            grid_width.append(max(all_lon) - min(all_lon))
            grid_height.append(max(all_lat) - min(all_lat))
        except Exception as e:
            continue
    grid_width = np.median(grid_width)
    grid_height = np.median(grid_height)

    global_min_col = min([extract_col_id(col) for col in smallest_column_per_row.values()])
    global_min_row = min([extract_row_id(row) for row in smallest_column_per_row.keys()])
    num_rows = len(smallest_column_per_row)
    num_cols = len(largest_row_per_column)

    for index, row in data_df_geo.iterrows():  # Assuming data_df_geo contains the geometries
        geometry = row['geometry']
        if geometry.is_empty:
            continue

        # Get the centroid of the geometry
        centroid = geometry.centroid
        y_center, x_center = centroid.y, centroid.x
        lon, lat = transformer.transform(x_center, y_center)

        # Get the row and column labels
        row_label = row['Crossmodel'][:4]
        col_label = row['Crossmodel'][4:]

        # Show all rows in the left-most non-NaN column
        if col_label == smallest_column_per_row[row_label]:
            folium.Marker(
                location=[lat, lon - 0.85 * grid_width],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 12px; color: red; font-weight: bold;">{row_label}</div>'
                )
            ).add_to(m)

        # Show all columns in the bottom-most non-NaN rows
        if row_label == largest_row_per_column[col_label]:
            # if it is the leftmost column, always print the column number at the top, avoiding overlapping with the row number
            if extract_row_id(col_label) == global_min_col:
                which_row = extract_row_id(row_label) - global_min_row
                if which_row > 1:   # would cause too much shifts or overlaps
                    continue
                else:
                    folium.Marker(
                        location=[lat + (which_row + 0.75) * grid_height, lon - 0.25 * grid_width],
                        icon=folium.DivIcon(
                            html=f'<div style="font-size: 12px; color: red; font-weight: bold;">{col_label}</div>'
                        )
                    ).add_to(m)
            else:
                folium.Marker(
                    location=[lat + 0.75 * grid_height, lon - 0.25 * grid_width],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size: 12px; color: red; font-weight: bold;">{col_label}</div>'
                    )
                ).add_to(m)

    return m


def overlay_heatmap_on_map(data_df, matrix, title, time_period, cell_geometries, color_norm, center_lat, center_lon, size_km=64, alpha=True, output_path="heatmap_map.png", verbose=False):
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
    x_columns = [col for col in data_df_geo.columns if col.endswith('_x')]
    for x_col in x_columns:
        base_col = x_col[:-2]  # Remove the '_x' suffix to get the base name
        y_col = f"{base_col}_y"

        data_df_geo[base_col] = data_df_geo[x_col]  # Or data_df_geo[y_col] if preferred
        data_df_geo.drop(columns=[x_col], inplace=True)
        if y_col in data_df_geo.columns:
            data_df_geo.drop(columns=[y_col], inplace=True)
    data_df_geo = data_df_geo.loc[:, ~data_df_geo.columns.duplicated()]
    # print('data_df_geo.columns', data_df_geo.columns)
    # if 'hist_x' in data_df_geo.columns and 'hist_y' in data_df_geo.columns:
    #     data_df_geo['hist'] = data_df_geo['hist_x']  # Or choose 'hist_y'
    #     data_df_geo.drop(columns=['hist_x', 'hist_y'], inplace=True)
    # data_df_geo = data_df_geo.loc[:, ~data_df_geo.columns.duplicated()]

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
                'weight': 0.8,
                'fillOpacity': 0.5,
            },
            tooltip=folium.features.GeoJsonTooltip(fields=['Crossmodel', col_name]),
        )
    )
    m = add_crossmodel_indices_on_map(data_df_geo, m)
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
    angle = find_grid_angle(screenshot)
    width, height = screenshot.size

    # Add title
    title_height = 30  # Space for title
    new_image = Image.new("RGB", (width, height + title_height), "white")
    new_image.paste(screenshot, (0, title_height))
    draw = ImageDraw.Draw(new_image)
    font = ImageFont.truetype("Arial_Black.ttf", 20)
    text_bbox = draw.textbbox((0, 0), title, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (width - text_width) // 2
    text_y = 0  # Padding from top
    draw.text((text_x, text_y), title, fill=(0, 0, 0), font=font)
    new_image.save(output_path)  # Overwrite the original image**

    if verbose:
        print(f"Map saved as image: {output_path} with size {width}x{height} pixels")
    driver.quit()

    return new_image, width, height, angle


def visualize_heatmap(matrix, title, color_norm, output_path="heatmap", verbose=False):
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
    plt.title(title.capitalize() + ' heatmap with row and column indices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()

    heatmap = Image.open(output_path)
    if verbose:
        print(f"Heatmap saved as: {output_path}")
    return heatmap, colormap, norm



def visualize_grids(data_df, matrix, title, time_period, cell_geometries, color_norm, center_lat, center_lon, size_km=64, output_path="heatmap", verbose=False):
    """
    Overlay a heatmap from a matrix onto a real map centered at the given latitude and longitude, and save as an image.
    """
    # Normalize the matrix for color mapping
    heatmap, colormap, norm = visualize_heatmap(matrix, title, color_norm, output_path=f"{output_path}.png", verbose=verbose)

    # Draw the final image with transparency on maps
    overlay_path = f"{output_path[:-1]}_overlay{output_path[-1]}.png"
    overlay, overlay_width, overlay_height, angle = overlay_heatmap_on_map(data_df, matrix, title, time_period, cell_geometries, color_norm, center_lat, center_lon, size_km, alpha=True, output_path=overlay_path, verbose=verbose)

    return heatmap, overlay, overlay_path, overlay_width, overlay_height, angle
