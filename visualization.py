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
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont
import folium
from folium.raster_layers import ImageOverlay
from folium import Html, Element
import geopandas as gpd
from shapely.geometry import Point
import branca.colormap as cm
from pyproj import Transformer
from mpl_toolkits.axes_grid1 import make_axes_locatable

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

import ocr


def find_grid_angle_and_bounding_box(image):
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
    # print("Detected line angle:", angle)

    heatmap_rect = find_largest_white_component_and_bounding_box(edge, -angle)

    return angle, heatmap_rect


def order_box_points(box_points):
    """
    Orders the four corner points of the rotated bounding box in the order:
    [top_left, top_right, lower_right, lower_left]
    """
    # Sort points based on y-coordinates (top-most first)
    box_points = sorted(box_points, key=lambda p: p[1])

    # Top two points (sort by x to get leftmost as top-left)
    top_left, top_right = sorted(box_points[:2], key=lambda p: p[0])
    # Bottom two points (sort by x to get leftmost as bottom-left)
    lower_left, lower_right = sorted(box_points[2:], key=lambda p: p[0])

    return np.array([top_left, top_right, lower_right, lower_left])


def find_largest_white_component_and_bounding_box(image, angle):
    """
    Finds the largest connected white component in a grayscale image and computes its bounding box.
    Then, it redefines the bounding box using the same center and edge lengths but with the angle set
    to the provided input_angle. The resulting box is drawn and saved on the image.

    :param image: NumPy array or PIL image.
    :param input_angle: The angle (in degrees) to which the bounding box should be rotated.
    :return: Ordered bounding box points (as an array) and the used angle.
    """
    # Convert PIL Image to NumPy array (if necessary) and to grayscale
    if not isinstance(image, np.ndarray):
        image = np.array(image.convert("L"))

    # Threshold the image to get a binary mask
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find contours of the white connected components
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours are found
    if not contours:
        raise ValueError("No contours found in the image.")

    # Find the largest contour based on area
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the largest component (if needed for further processing)
    largest_component_mask = np.zeros_like(thresh)
    cv2.drawContours(largest_component_mask, [largest_contour], -1, color=255, thickness=-1)

    # Compute the rotated bounding box using the largest contour
    rect = cv2.minAreaRect(largest_contour)
    center, size, _ = rect  # ignore the angle computed by minAreaRect

    # Create a new rectangle with the same center and size but with the angle set to input_angle.
    new_rect = (center, size, angle)

    # Get the box points for the new rotated rectangle and convert them to integers.
    box_points = cv2.boxPoints(new_rect)
    box_points = np.intp(box_points)
    ordered_box_points = order_box_points(box_points)

    # Draw the new bounding box on the original image (convert to BGR for visualization)
    annotated_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(annotated_image, [ordered_box_points], 0, (0, 255, 0), 2)

    # Save the result
    cv2.imwrite('box_of_heatmap.jpg', annotated_image)

    return ordered_box_points


def add_crossmodel_indices_on_map(data_df_geo, m, col_name=None):
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    row_leftmost = {}  # Store the leftmost geometry per row
    col_topmost = {}   # Store the topmost geometry per column

    # Identify the extreme positions for row and column labels
    for _, data_row in data_df_geo.iterrows():
        geometry = data_row['geometry']
        if geometry.is_empty:
            continue

        crossmodel = data_row['Crossmodel']
        row_label, col_label = crossmodel[:4], crossmodel[4:]

        centroid = geometry.centroid
        x_center, y_center = centroid.x, centroid.y

        # Track leftmost position for each row
        if row_label not in row_leftmost or x_center < row_leftmost[row_label][0]:
            row_leftmost[row_label] = (x_center, y_center, geometry.bounds[0])  # Store leftmost x-bound

        # Track topmost position for each column
        if col_label not in col_topmost or y_center > col_topmost[col_label][1]:
            col_topmost[col_label] = (x_center, y_center, geometry.bounds[3])  # Store topmost y-bound

    # Place row labels at the leftmost position of the leftmost grid
    for r_label, (x, y, x_leftmost) in row_leftmost.items():
        lon, lat = transformer.transform(x_leftmost, y)  # Use left-most x
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size: 12px; color: red; font-weight: bold;">{r_label}</div>'
            )
        ).add_to(m)

    # Place column labels at the topmost position of the topmost grid
    for c_label, (x, y, y_topmost) in col_topmost.items():
        lon, lat = transformer.transform(x, y_topmost)  # Use top-most y
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size: 12px; color: red; font-weight: bold;">{c_label}</div>'
            )
        ).add_to(m)

    if col_name:
        all_data = []
        # Place numerical values at the correct location in black
        for _, row in data_df_geo.iterrows():
            geometry = row['geometry']
            if geometry.is_empty:
                continue

            centroid = geometry.centroid
            x_center, y_center = centroid.x, centroid.y
            lon, lat = transformer.transform(x_center, y_center)  # Convert projection

            value = row[col_name]
            all_data.append(value)
            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 10px; color: black;">{value:.2f}</div>'
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
    # angle = find_grid_angle(screenshot)
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

    return new_image, width, height


def visualize_heatmap(data_df, matrix, title, time_period, cell_geometries, color_norm,
                      center_lat, center_lon, size_km=64, alpha=True, output_path="heatmap.png", add_text=False, verbose=False):
    if add_text:
        output_path = output_path[:-4] + "_with_text" + output_path[-4:]

    # Extract the column corresponding to the time_period
    variable_columns = [col for col in data_df.columns if time_period == col]
    variable_df = data_df[['Crossmodel'] + variable_columns]
    col_name = variable_columns[0]

    # Merge with geometries
    variable_df = variable_df.merge(cell_geometries, on='Crossmodel', how='inner')
    data_df_geo = gpd.GeoDataFrame(
        data_df.merge(variable_df, on='Crossmodel', how='inner', suffixes=('_x', '_y'))
    )

    # Remove duplicated columns (those ending with _x or _y)
    x_columns = [col for col in data_df_geo.columns if col.endswith('_x')]
    for x_col in x_columns:
        base_col = x_col[:-2]
        y_col = f"{base_col}_y"
        data_df_geo[base_col] = data_df_geo[x_col]
        data_df_geo.drop(columns=[x_col], inplace=True)
        if y_col in data_df_geo.columns:
            data_df_geo.drop(columns=[y_col], inplace=True)
    data_df_geo = data_df_geo.loc[:, ~data_df_geo.columns.duplicated()]

    # Check if required fields exist
    required_fields = ['Crossmodel', col_name, 'geometry']
    for field in required_fields:
        if field not in data_df_geo.columns:
            raise ValueError(f"Missing required field '{field}' in data_df_geo.")

    # Define the colormap based on color_norm values
    colormap = cm.LinearColormap(
        colors=['#276AAE', '#7FB6D5', '#E6E1DE', '#F39F7B', '#B41D2E'],
        vmin=color_norm.vmin,
        vmax=color_norm.vmax
    )

    # Function to convert numerical values to color strings
    def value_to_color(value):
        if value is None or np.isnan(value):
            return 'rgba(0,0,0,0)'
        return colormap(value)

    data_df_geo['color'] = data_df_geo[col_name].apply(value_to_color)

    # Create a folium map with no base tiles so only the heatmap polygons show
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles=None)
    m.add_child(
        folium.features.GeoJson(
            data_df_geo,
            style_function=lambda feature: {
                'fillColor': feature['properties']['color'],
                'color': 'black',  # Border color for polygons
                'weight': 0.8,
                'fillOpacity': 0.5,
            },
            tooltip=folium.features.GeoJsonTooltip(fields=['Crossmodel', col_name]),
        )
    )
    if add_text:
        m = add_crossmodel_indices_on_map(data_df_geo, m, col_name)
    else:
        m = add_crossmodel_indices_on_map(data_df_geo, m)
    colormap.caption = "Values"
    m.add_child(colormap)

    # Save the folium map as an HTML file (use an identifier to avoid conflicts)
    temp_html = f"temp_heatmap_{output_path[-5]}.html"
    m.save(temp_html)

    # Use Selenium to capture an image of the map
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(service=ChromeService(), options=chrome_options)

    # Calculate window size based on the matrix dimensions and a km-to-pixel ratio
    rows, cols = matrix.shape
    km_per_block = 12
    km_to_pixel_ratio = 8
    window_width = int(cols * km_per_block * km_to_pixel_ratio)
    window_height = int(rows * km_per_block * km_to_pixel_ratio)
    driver.set_window_size(window_width, window_height)

    driver.get("file://" + os.path.abspath(temp_html))
    sleep(3)  # Allow time for the map to fully load
    driver.save_screenshot(output_path)
    screenshot = Image.open(output_path)
    angle, heatmap_rect = find_grid_angle_and_bounding_box(screenshot)  # Assuming this function is defined elsewhere
    width, height = screenshot.size

    # Optionally add a title bar to the image (if desired)
    title_height = 30  # Height for the title area
    new_image = Image.new("RGB", (width, height + title_height), "white")
    new_image.paste(screenshot, (0, title_height))
    draw = ImageDraw.Draw(new_image)
    font = ImageFont.truetype("Arial_Black.ttf", 20)
    text_bbox = draw.textbbox((0, 0), title, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (width - text_width) // 2
    draw.text((text_x, 0), title, fill=(0, 0, 0), font=font)
    new_image.save(output_path)  # Save the final image

    if verbose:
        print(f"Heatmap image saved as: {output_path} with size {width}x{height} pixels")
    driver.quit()

    return new_image, angle, heatmap_rect


def visualize_grids(question_dir, data_df, matrix, title, time_period, cell_geometries, color_norm, center_lat, center_lon, size_km=64, output_path="heatmap", verbose=False):
    """
    Overlay a heatmap from a matrix onto a real map centered at the given latitude and longitude, and save as an image.
    """
    # Normalize the matrix for color mapping
    # heatmap, colormap, norm = visualize_heatmap(matrix, title, color_norm, output_path=os.path.join(question_dir, f"{output_path}.png"), verbose=verbose)
    # heatmap = visualize_heatmap(data_df, time_period, cell_geometries, color_norm, title, output_path=os.path.join(question_dir, f"{output_path}.png"), verbose=verbose)
    heatmap, angle, heatmap_rect = visualize_heatmap(data_df, matrix, title, time_period, cell_geometries, color_norm, center_lat, center_lon, size_km, alpha=True,
                                output_path=os.path.join(question_dir, f"{output_path}.png"), add_text=False, verbose=verbose)
    heatmap_with_text, _, _ = visualize_heatmap(data_df, matrix, title, time_period, cell_geometries, color_norm, center_lat, center_lon, size_km, alpha=True,
                                output_path=os.path.join(question_dir, f"{output_path}.png"), add_text=True, verbose=verbose)

    # Draw the final image with transparency on maps
    overlay_path = f"{output_path[:-1]}_overlay{output_path[-1]}.png"
    overlay, overlay_width, overlay_height = overlay_heatmap_on_map(data_df, matrix, title, time_period, cell_geometries,
                                                        color_norm, center_lat, center_lon, size_km, alpha=True, output_path=os.path.join(question_dir, overlay_path), verbose=verbose)

    return heatmap, heatmap_with_text, overlay, overlay_path, overlay_width, overlay_height, angle, heatmap_rect
