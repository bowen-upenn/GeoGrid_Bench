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

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

import ocr


def overlay_heatmap_on_map(matrix, variable_name, center_lat, center_lon, size_km=64, alpha=True, output_path="heatmap_map.png", verbose=False):
    # Use visualize_heatmap to create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    mid_point = (matrix.max().max() + matrix.min().min()) / 2
    norm = TwoSlopeNorm(vmin=matrix.min().min(), vmax=matrix.max().max(), vcenter=mid_point)
    colormap = plt.get_cmap('coolwarm')
    heatmap = ax.imshow(matrix, cmap=colormap, norm=norm)

    # Add column and row indices as red bold text
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels([f'C{col}' for col in matrix.columns], fontsize=15, color='red', fontweight='bold')
    ax.set_yticklabels([f'R{row}' for row in matrix.index], fontsize=15, color='red', fontweight='bold')

    # Save the figure to a buffer
    buf = io.BytesIO()
    plt.title(variable_name.capitalize() + ' heatmap with row and column indices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    # Load the image from buffer and make it semi-transparent
    image = Image.open(buf)
    image = image.convert("RGBA")
    alpha = 100 if alpha else 255
    for y in range(image.height):
        for x in range(image.width):
            r, g, b, a = image.getpixel((x, y))
            if a > 0 and ((r, g, b) != (255, 255, 255) and (r, g, b) != (255, 0, 0) and (r, g, b) != (0, 0, 0)):    # heatmaps
                image.putpixel((x, y), (r, g, b, alpha))
            elif (r, g, b) == (255, 255, 255):  # white background
                image.putpixel((x, y), (r, g, b, 200))
            elif (r, g, b) == (255, 0, 0) or (r, g, b) == (0, 0, 0):    # red and black text
                image.putpixel((x, y), (r, g, b, 255))

    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    img_url = f"data:image/png;base64,{img_base64}"

    # Set bounds for the overlay
    half_size_deg = (size_km / 111) / 2  # Approximate conversion km -> degrees
    bounds = [[center_lat - half_size_deg, center_lon - half_size_deg],
              [center_lat + half_size_deg, center_lon + half_size_deg]]

    # Create the map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    ImageOverlay(
        name="Heatmap Overlay",
        image=img_url,
        bounds=bounds,
        opacity=0.9
    ).add_to(m)

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


def visualize_heatmap(matrix, variable_name, output_path="heatmap", verbose=False):
    # Normalize the matrix for color mapping
    norm = Normalize(vmin=matrix.min().min(), vmax=matrix.max().max())
    colormap = plt.get_cmap('coolwarm')

    # Create the heatmap visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = ax.imshow(matrix, cmap=colormap, norm=norm)

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



def visualize_grids(matrix, variable_name, center_lat, center_lon, size_km=64, output_path="heatmap", verbose=False):
    """
    Overlay a heatmap from a matrix onto a real map centered at the given latitude and longitude, and save as an image.
    """
    # Normalize the matrix for color mapping
    heatmap, colormap, norm = visualize_heatmap(matrix, variable_name, output_path=f"{output_path}.png", verbose=verbose)

    # Draw the final image with transparency on maps
    overlay_path = f"{output_path[:-1]}_overlay{output_path[-1]}.png"
    overlay, overlay_width, overlay_height = overlay_heatmap_on_map(matrix, variable_name, center_lat, center_lon, size_km, alpha=True, output_path=overlay_path, verbose=verbose)

    return heatmap, overlay, overlay_path, overlay_width, overlay_height
