#!/usr/bin/env python3
"""
Cities Skylines 2 Heightmap Generator

This script downloads and processes elevation data to create heightmaps compatible with 
Cities Skylines 2. It:
1. Takes a center point of interest (latitude/longitude)
2. Queries available elevation data from USGS 3DEP
3. Downloads the best quality data
4. Processes it into heightmaps conforming to CS2 requirements:
   - Base heightmap (4096x4096)
   - Extended worldmap (4096x4096 with the central 1024x1024 matching the base heightmap)
"""

import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from PIL import Image
import argparse
from scipy import ndimage
import time
import re

def calculate_bounding_box(center_lat, center_lon, width_km=32.768, height_km=32.768):
    """
    Calculate a bounding box around a center point.
    CS2 map size is 4096 pixels at 8m/pixel = 32.768 km total width/height
    
    Args:
        center_lat (float): Center latitude
        center_lon (float): Center longitude
        width_km (float): Width of the bounding box in kilometers
        height_km (float): Height of the bounding box in kilometers
        
    Returns:
        tuple: Bounding box as (min_lon, min_lat, max_lon, max_lat)
    """
    # Approximate degrees per km (varies by latitude)
    lat_degree_per_km = 1/111.32  # roughly 1 degree = 111.32 km
    lon_degree_per_km = 1/(111.32 * np.cos(np.radians(center_lat)))  # adjusts for latitude
    
    # Calculate offsets
    lat_offset = (height_km/2) * lat_degree_per_km
    lon_offset = (width_km/2) * lon_degree_per_km
    
    # Calculate bounding box
    min_lat = center_lat - lat_offset
    max_lat = center_lat + lat_offset
    min_lon = center_lon - lon_offset
    max_lon = center_lon + lon_offset
    
    return min_lon, min_lat, max_lon, max_lat

def query_available_elevation_data(bbox):
    """
    Query the 3DEP service for available data layers.
    
    Args:
        bbox (tuple): Bounding box as (min_lon, min_lat, max_lon, max_lat)
        
    Returns:
        dict: Information about available elevation data
    """
    print("Searching for available elevation data...")
    
    # Define resolution layers to check in order of preference
    resolution_layers = [
        {"name": "1 meter", "layer": "0", "resolution": 1},
        {"name": "3 meter", "layer": "1", "resolution": 3},
        {"name": "5 meter", "layer": "2", "resolution": 5},
        {"name": "10 meter", "layer": "3", "resolution": 10},
        {"name": "30 meter", "layer": "4", "resolution": 30},
        {"name": "60 meter", "layer": "5", "resolution": 60}
    ]
    
    # Also try direct access to the 3DEP elevation service
    elevation_service_url = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/"
    params = {"f": "json"}
    response = requests.get(elevation_service_url, params=params)
    elevation_info = response.json()
    print(f"3DEP Elevation Service available: {'capabilities' in elevation_info}")
    
    # Check each resolution layer
    available_data = []
    arcgis_url = "https://index.nationalmap.gov/arcgis/rest/services/3DEPElevationIndex/MapServer/find"
    
    for res_layer in resolution_layers:
        params = {
            "searchText": res_layer["name"],
            "contains": "true",
            "searchFields": "ProjectName",
            "sr": "4326",  # WGS84 coordinate system
            "layers": res_layer["layer"],
            "returnGeometry": "true",
            "f": "json",
            "geometryType": "esriGeometryEnvelope",
            "spatialRel": "esriSpatialRelIntersects",
            "geometry": json.dumps({"xmin": bbox[0], "ymin": bbox[1], "xmax": bbox[2], "ymax": bbox[3]})
        }
        
        print(f"Checking for {res_layer['name']} resolution data...")
        response = requests.get(arcgis_url, params=params)
        data = response.json()
        
        if 'results' in data and len(data['results']) > 0:
            print(f"Found {len(data['results'])} datasets with {res_layer['name']} resolution")
            # Add resolution info to the results
            for result in data['results']:
                result['resolution'] = res_layer['resolution']
            available_data.append({
                "resolution": res_layer['resolution'],
                "name": res_layer['name'],
                "results": data['results']
            })
    
    # Check if any data was found
    if not available_data:
        print("No elevation data found through the 3DEP index. Will try direct elevation service.")
        # We can still try to get data directly from elevation service
        return {"direct_service": True, "results": []}
    
    # Return the best resolution data we found
    return {"direct_service": False, "data": available_data}

def download_elevation_data(elevation_data_info, bbox, output_dir="elevation_output"):
    """
    Download elevation data based on the query results.
    
    Args:
        elevation_data_info (dict): Information about available elevation data
        bbox (tuple): Bounding box as (min_lon, min_lat, max_lon, max_lat)
        output_dir (str): Directory to save downloaded data
        
    Returns:
        tuple: (path to downloaded file, source type, resolution)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # First try direct download from 3DEP elevation service
    def try_direct_elevation_service():
        print("Attempting direct download from 3DEP elevation service...")
        elevation_service_url = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
        
        params = {
            "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "bboxSR": 4326,  # WGS84
            "size": "4096,4096",
            "imageSR": 4326,
            "format": "tiff",
            "pixelType": "F32",
            "interpolation": "RSP_BilinearInterpolation",
            "f": "json"
        }
        
        # Get the image data info
        response = requests.get(elevation_service_url, params=params)
        image_info = response.json()
        
        if 'href' in image_info:
            # Download the image
            img_url = image_info['href']
            print(f"Found direct download URL: {img_url}")
            img_response = requests.get(img_url)
            
            if img_response.status_code == 200:
                raw_tiff_path = os.path.join(output_dir, "raw_elevation.tiff")
                with open(raw_tiff_path, 'wb') as f:
                    f.write(img_response.content)
                print(f"Downloaded elevation data to {raw_tiff_path}")
                return raw_tiff_path, 'direct', None  # Resolution unknown when using direct service
        
        print("Direct download failed or no data available.")
        return None, None, None
    
    # Try indexed data download if available
    if not elevation_data_info["direct_service"] and "data" in elevation_data_info and elevation_data_info["data"]:
        # Get the best (lowest value) resolution data
        best_data = min(elevation_data_info["data"], key=lambda x: x["resolution"])
        resolution = best_data["resolution"]
        print(f"Using best available resolution: {best_data['name']} ({resolution}m)")
        
        if best_data["results"]:
            # Get the first result (they're sorted by relevance)
            result = best_data["results"][0]
            # Here we would extract download URL from the result and download the data
            # This is complex and depends on the specific format of the result
            print("Indexed data format requires custom extraction. Falling back to direct service.")
            return try_direct_elevation_service()
    
    # If no indexed data or extraction failed, try direct service
    return try_direct_elevation_service()

def interpolate_nodata(data, mask):
    """
    Interpolate no-data values in elevation data.
    
    Args:
        data (numpy.ndarray): The elevation data
        mask (numpy.ndarray): Boolean mask of no-data values
        
    Returns:
        numpy.ndarray: The elevation data with interpolated values
    """
    # Simple interpolation: replace no-data with the mean of non-masked neighbors
    # Create a copy to avoid modifying the original during interpolation
    filled = data.copy()
    # Use a mean filter to compute neighbor values
    temp = ndimage.gaussian_filter(np.where(~mask, filled, np.nan), sigma=2)
    # Only replace the masked values
    filled[mask] = temp[mask]
    # If there are still NaNs, use the overall mean of valid data
    if np.isnan(filled).any():
        valid_mean = np.nanmean(filled)
        filled[np.isnan(filled)] = valid_mean
    return filled

def get_location_name(lat, lon):
    """
    Get the name of a location from OpenStreetMap Nominatim API.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        str: Location name or None if not found
    """
    print("Looking up location name...")
    try:
        # Add a user agent to comply with Nominatim usage policy
        headers = {
            'User-Agent': 'CS2HeightmapGenerator/1.0',
        }
        
        # Query the Nominatim API
        params = {
            'format': 'json',
            'lat': lat,
            'lon': lon,
            'zoom': 12,  # Adjust zoom level for appropriate detail
            'addressdetails': 1,
        }
        
        # Add a small delay to respect rate limits
        response = requests.get('https://nominatim.openstreetmap.org/reverse', params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if 'name' in data:
                return data['name']
            elif 'address' in data:
                # Build name from address components, prioritizing more specific location names
                addr = data['address']
                for key in ['city', 'town', 'village', 'hamlet', 'suburb', 'county', 'state']:
                    if key in addr:
                        return addr[key]
                
                # If no specific component found, use display name
                if 'display_name' in data:
                    # Extract first part of display name (typically most specific)
                    return data['display_name'].split(',')[0]
        
        # If we couldn't find a suitable name
        print("Couldn't find specific location name.")
        return None
    
    except Exception as e:
        print(f"Error getting location name: {str(e)}")
        return None

def sanitize_filename(name):
    """
    Sanitize a string to be used as a filename.
    
    Args:
        name (str): Input string
        
    Returns:
        str: Sanitized string
    """
    if not name:
        return "unknown"
    
    # Replace invalid characters
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    # Trim whitespace and limit length
    name = name.strip()[:50]
    return name

def process_heightmap(raw_tiff_path, output_dir="elevation_output", location_name=None, max_height=4096, create_worldmap=True):
    """
    Process the raw elevation data into a CS2 compatible heightmap.
    
    Args:
        raw_tiff_path (str): Path to the raw elevation TIFF file
        output_dir (str): Directory to save processed heightmaps
        location_name (str): Name of the location (for file naming)
        max_height (int): Maximum height scale for CS2
        create_worldmap (bool): Whether to create an extended worldmap
        
    Returns:
        dict: Information about the processed heightmap
    """
    if not raw_tiff_path or not os.path.exists(raw_tiff_path):
        print("No raw data to process!")
        return None
    
    try:
        # Generate filename base from location name if available
        if location_name:
            filename_base = sanitize_filename(location_name)
        else:
            filename_base = "heightmap"
        
        # Read the raw elevation data
        with rasterio.open(raw_tiff_path) as src:
            print(f"Raster info: {src.width}x{src.height}, {src.count} bands, dtype: {src.dtypes[0]}")
            
            elevation_data = src.read(1)  # Read the first band
            
            # Check for no-data values
            no_data = src.nodata
            if no_data is not None:
                print(f"No-data value: {no_data}")
                # Replace no-data with neighboring values
                mask = elevation_data == no_data
                if mask.any():
                    print(f"Found {mask.sum()} no-data pixels. Interpolating...")
                    elevation_data = interpolate_nodata(elevation_data, mask)
            
            # Handle any remaining NaN or infinite values
            if np.isnan(elevation_data).any() or np.isinf(elevation_data).any():
                print("Found NaN or Inf values. Replacing with valid neighbor values...")
                mask = np.logical_or(np.isnan(elevation_data), np.isinf(elevation_data))
                elevation_data = interpolate_nodata(elevation_data, mask)
            
            # Get statistics for normalization
            min_elev = np.min(elevation_data)
            max_elev = np.max(elevation_data)
            elev_range = max_elev - min_elev
            
            if elev_range == 0:
                print("Warning: Flat terrain detected (no elevation difference)")
                # Add small random variations for visual interest
                elevation_data = elevation_data + np.random.rand(*elevation_data.shape) * 0.1
                min_elev = np.min(elevation_data)
                max_elev = np.max(elevation_data)
                elev_range = max_elev - min_elev
            
            print(f"Elevation range: {min_elev:.2f}m to {max_elev:.2f}m")
            print(f"Total elevation difference: {elev_range:.2f}m")
            
            # Normalize to 0-1 range
            normalized = (elevation_data - min_elev) / elev_range
            
            # Scale to 16-bit (0-65535) range for 16-bit grayscale output
            scaled = (normalized * 65535).astype(np.uint16)
            
            # Save as 16-bit PNG
            png_path = os.path.join(output_dir, f"{filename_base}.png")
            Image.fromarray(scaled).save(png_path)
            print(f"Saved 16-bit PNG heightmap to {png_path}")
            
            # Save as 16-bit TIFF
            tiff_path = os.path.join(output_dir, f"{filename_base}.tiff")
            
            # Copy the metadata from source
            meta = src.meta.copy()
            meta.update({
                'dtype': 'uint16',
                'width': 4096,
                'height': 4096,
                'nodata': None  # We've handled no-data values
            })
            
            with rasterio.open(tiff_path, 'w', **meta) as dst:
                dst.write(scaled, 1)
            print(f"Saved 16-bit TIFF heightmap to {tiff_path}")
            
            # Generate worldmap if requested
            worldmap_paths = None
            if create_worldmap:
                # Get the center coordinates of the source data
                # This helps us calculate a larger bounding box for the worldmap
                with rasterio.open(raw_tiff_path) as src:
                    # Get the bounds of the image (in the CRS of the image)
                    bounds = src.bounds
                    # Get the transform to convert from image coordinates to CRS coordinates
                    transform = src.transform
                    # Calculate center point
                    center_x = (bounds.left + bounds.right) / 2
                    center_y = (bounds.top + bounds.bottom) / 2
                    
                    # For worldmap we need latitude/longitude
                    if src.crs.to_epsg() == 4326:  # If already in WGS84
                        center_lon, center_lat = center_x, center_y
                    else:
                        # We'd need to reproject to get lat/lon
                        # For simplicity, we'll use the lat/lon from args
                        # This would be where we'd use pyproj to transform coordinates
                        # but for simplicity we'll extract from args later
                        center_lat, center_lon = None, None
                
                worldmap_paths = generate_worldmap(scaled, meta, output_dir, center_lat, center_lon, min_elev, max_elev, filename_base)
                if worldmap_paths:
                    print(f"Worldmap generated successfully:")
                    print(f"- PNG: {worldmap_paths['png']}")
                    print(f"- TIFF: {worldmap_paths['tiff']}")
            
            return {
                "png": png_path, 
                "tiff": tiff_path, 
                "min_elev": min_elev, 
                "max_elev": max_elev, 
                "range": elev_range,
                "worldmap": worldmap_paths,
                "location_name": location_name
            }
    except Exception as e:
        print(f"Error processing elevation data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_worldmap(base_heightmap, meta, output_dir, center_lat=None, center_lon=None, min_elev=None, max_elev=None, filename_base="heightmap"):
    """
    Generate a worldmap heightmap where the central 1024x1024 matches the base heightmap.
    Downloads real elevation data for a larger area when possible, using lower resolution 
    for better performance since this area is outside the playable zone.
    
    Args:
        base_heightmap (numpy.ndarray): The base heightmap data (4096x4096)
        meta (dict): Rasterio metadata for the base heightmap
        output_dir (str): Directory to save the worldmap
        center_lat (float): Center latitude (if available)
        center_lon (float): Center longitude (if available)
        min_elev (float): Minimum elevation from base heightmap
        max_elev (float): Maximum elevation from base heightmap
        filename_base (str): Base filename to use for output files
        
    Returns:
        dict: Paths to the worldmap files
    """
    print("Generating extended worldmap...")
    
    # Extract the center 1024x1024 of the base_heightmap
    print("1. Extracting central region of base heightmap...")
    h_center_start = (base_heightmap.shape[0] - 1024) // 2
    h_center_end = h_center_start + 1024
    w_center_start = (base_heightmap.shape[1] - 1024) // 2
    w_center_end = w_center_start + 1024
    
    center_heightmap = base_heightmap[h_center_start:h_center_end, w_center_start:w_center_end]
    
    # Try to download real-world data for the larger worldmap area if coordinates are available
    worldmap = None
    raw_worldmap_path = None
    
    if center_lat is not None and center_lon is not None:
        print("2. Attempting to download real elevation data for extended worldmap...")
        try:
            # Calculate a larger bounding box for worldmap (4x larger area)
            # CS2 worldmap covers 57.344 x 57.344 km
            worldmap_bbox = calculate_bounding_box(center_lat, center_lon, 
                                                  width_km=57.344, 
                                                  height_km=57.344)
            
            # Download elevation data for the larger area
            # Note: For the worldmap, we can use lower resolution since it's outside the playable area
            # We'll request fewer pixels (2048x2048 instead of 4096x4096) and then upsample
            elevation_service_url = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
            
            params = {
                "bbox": f"{worldmap_bbox[0]},{worldmap_bbox[1]},{worldmap_bbox[2]},{worldmap_bbox[3]}",
                "bboxSR": 4326,  # WGS84
                "size": "2048,2048",  # Lower resolution for the worldmap
                "imageSR": 4326,
                "format": "tiff",
                "pixelType": "F32",
                "interpolation": "RSP_BilinearInterpolation",
                "f": "json"
            }
            
            # Get the image data info
            response = requests.get(elevation_service_url, params=params)
            image_info = response.json()
            
            if 'href' in image_info:
                # Download the image
                img_url = image_info['href']
                print(f"Found direct download URL for worldmap: {img_url}")
                img_response = requests.get(img_url)
                
                if img_response.status_code == 200:
                    # Create a unique filename using timestamp to avoid conflicts
                    import uuid
                    unique_id = str(uuid.uuid4())[:8]
                    raw_worldmap_path = os.path.join(output_dir, f"raw_worldmap_{unique_id}.tiff")
                    
                    # Save the raw worldmap data
                    with open(raw_worldmap_path, 'wb') as f:
                        f.write(img_response.content)
                    
                    # Process the worldmap data in a way that ensures file handles are closed
                    worldmap_data = None
                    with rasterio.open(raw_worldmap_path) as src:
                        worldmap_data = src.read(1).copy()  # Make a copy to ensure we're not holding the file
                        
                        # Get no-data value before closing
                        no_data = src.nodata
                    
                    # Now that the file is closed, process the data
                    if worldmap_data is not None:
                        # Handle no-data values
                        if no_data is not None:
                            mask = worldmap_data == no_data
                            if mask.any():
                                print(f"Interpolating {mask.sum()} no-data pixels in worldmap...")
                                worldmap_data = interpolate_nodata(worldmap_data, mask)
                        
                        # Handle NaN or inf values
                        if np.isnan(worldmap_data).any() or np.isinf(worldmap_data).any():
                            mask = np.logical_or(np.isnan(worldmap_data), np.isinf(worldmap_data))
                            worldmap_data = interpolate_nodata(worldmap_data, mask)
                        
                        # Upsample from 2048x2048 to 4096x4096
                        print("Upsampling worldmap data to full resolution...")
                        worldmap_data = ndimage.zoom(worldmap_data, 2, order=1)
                        
                        # Normalize the worldmap data using the same elevation range as the base heightmap
                        # This ensures consistent height representation between the two maps
                        if min_elev is not None and max_elev is not None:
                            # Clip the worldmap data to the same min/max as the base heightmap
                            # to ensure consistent scaling
                            worldmap_data = np.clip(worldmap_data, min_elev, max_elev)
                            worldmap_normalized = (worldmap_data - min_elev) / (max_elev - min_elev)
                        else:
                            # If we don't have the base heightmap range, normalize based on this data
                            wm_min = np.min(worldmap_data)
                            wm_max = np.max(worldmap_data)
                            worldmap_normalized = (worldmap_data - wm_min) / (wm_max - wm_min)
                        
                        # Scale to 16-bit
                        worldmap = (worldmap_normalized * 65535).astype(np.uint16)
                        print("Successfully downloaded and processed real elevation data for worldmap")
        except Exception as e:
            print(f"Error downloading worldmap data: {str(e)}")
            print("Falling back to procedural generation for worldmap")
            worldmap = None
        finally:
            # Clean up the temporary file in the finally block to ensure it happens
            try:
                if raw_worldmap_path and os.path.exists(raw_worldmap_path):
                    # Add a small delay to ensure file handles are closed
                    time.sleep(0.1)
                    os.remove(raw_worldmap_path)
                    print(f"Cleaned up temporary file: {raw_worldmap_path}")
            except Exception as e:
                print(f"Warning: Could not remove temporary file {raw_worldmap_path}: {str(e)}")
                print("This is not critical - continuing with worldmap generation")
    
    # If we couldn't download real data, generate a procedural worldmap
    if worldmap is None:
        print("2. Creating extended worldmap with procedural terrain...")
        
        # Create a new worldmap array as float32 to avoid casting issues during calculations
        worldmap_float = np.zeros((4096, 4096), dtype=np.float32)
        
        # Start with a low-resolution version of our base heightmap
        downsampled = ndimage.zoom(base_heightmap.astype(np.float32), 0.25, order=1)  # 4096 -> 1024
        
        # Upsample it back to full size (this creates a smoothed version)
        # This will serve as the base for our extended worldmap
        smooth_base = ndimage.zoom(downsampled, 4, order=1)  # 1024 -> 4096
        worldmap_float = smooth_base.copy()
        
        # Create distance-based mask for blending (1 at the edges, 0 in center)
        y, x = np.ogrid[:4096, :4096]
        center_y, center_x = 4096/2, 4096/2
        # Distance from center (normalized to 0-1)
        distance = np.sqrt(((y - center_y) / center_y) ** 2 + ((x - center_x) / center_x) ** 2)
        
        # Create a soft-edge square mask for the center region
        center_mask = np.zeros((4096, 4096), dtype=np.float32)
        center_mask[h_center_start:h_center_end, w_center_start:w_center_end] = 1.0
        
        # Create a smooth transition between center and surrounding area
        transition_width = 128
        center_mask_smooth = ndimage.gaussian_filter(center_mask, sigma=transition_width/2)
        
        # Normalize to 0-1 range
        center_mask_smooth = np.clip(center_mask_smooth, 0, 1)
        
        # Place the original center heightmap
        worldmap_float[h_center_start:h_center_end, w_center_start:w_center_end] = center_heightmap.astype(np.float32)
        
        # Add procedural terrain variation to areas outside the center
        # Get statistics from the base heightmap to inform our procedural generation
        base_mean = np.mean(base_heightmap)
        base_std = np.std(base_heightmap)
        
        # Generate terrain variations with multiple octaves of noise
        for octave in range(3):
            # Generate noise at appropriate scale for this octave
            size = 32 * (2 ** octave)  # 32, 64, 128
            noise_base = np.random.RandomState(octave + 1).rand(size, size)
            
            # Scale up to full size
            noise = ndimage.zoom(noise_base, 4096/size, order=3)
            
            # Apply gaussian smoothing
            sigma = 64 / (2 ** octave)
            noise = ndimage.gaussian_filter(noise, sigma=sigma)
            
            # Scale noise to match the range of the base heightmap
            noise = (noise - 0.5) * base_std * (0.5 ** octave)
            
            # Weight noise based on distance from center (more noise further from center)
            # Use inverse of center mask to only apply noise outside center
            noise_weight = (1.0 - center_mask_smooth) * (0.3 ** octave)
            worldmap_float += noise * noise_weight
        
        # Ensure smooth transition between center and surrounding regions
        # Use linear blending between center and smooth_base+noise
        blend_region = np.zeros_like(worldmap_float)
        blend_region[h_center_start:h_center_end, w_center_start:w_center_end] = center_heightmap.astype(np.float32)
        
        # Final worldmap is a blend of the original center and the procedural surrounding terrain
        worldmap_final = center_mask_smooth * blend_region + (1 - center_mask_smooth) * worldmap_float
        
        # Normalize to 16-bit range
        worldmap_min = np.min(worldmap_final)
        worldmap_max = np.max(worldmap_final)
        worldmap_normalized = (worldmap_final - worldmap_min) / (worldmap_max - worldmap_min)
        
        # Scale to 16-bit and convert to uint16
        worldmap = (worldmap_normalized * 65535).astype(np.uint16)
    
    print("3. Saving worldmap files...")
    # Save as 16-bit PNG
    worldmap_png_path = os.path.join(output_dir, f"{filename_base}_worldmap.png")
    Image.fromarray(worldmap).save(worldmap_png_path)
    
    # Save as 16-bit TIFF
    worldmap_tiff_path = os.path.join(output_dir, f"{filename_base}_worldmap.tiff")
    
    # Copy the metadata and update for worldmap
    worldmap_meta = meta.copy()
    
    with rasterio.open(worldmap_tiff_path, 'w', **worldmap_meta) as dst:
        dst.write(worldmap, 1)
    
    return {
        "png": worldmap_png_path,
        "tiff": worldmap_tiff_path
    }

def visualize_heightmaps(heightmap_info, center_lat, center_lon, output_dir="elevation_output"):
    """
    Create visualizations of the heightmaps.
    
    Args:
        heightmap_info (dict): Information about the heightmaps, including paths
        center_lat (float): Center latitude
        center_lon (float): Center longitude
        output_dir (str): Directory to save visualizations
    """
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Load the base heightmap
    base_heightmap = np.array(Image.open(heightmap_info['png']))
    
    # Get the location name for the title
    location_name = heightmap_info.get('location_name', '')
    location_title = f' ({location_name})' if location_name else ''
    
    # Plot the base heightmap
    im1 = ax1.imshow(base_heightmap, cmap='terrain')
    ax1.set_title(f'Base Heightmap{location_title}\nCentered at {center_lat}, {center_lon}')
    ax1.grid(alpha=0.3)
    plt.colorbar(im1, ax=ax1, label='Elevation (scaled to 16-bit)')
    
    # Load and plot worldmap if available
    if heightmap_info['worldmap']:
        worldmap = np.array(Image.open(heightmap_info['worldmap']['png']))
        im2 = ax2.imshow(worldmap, cmap='terrain')
        ax2.set_title(f'Extended Worldmap{location_title}\n(4x larger area)')
        ax2.grid(alpha=0.3)
        plt.colorbar(im2, ax=ax2, label='Elevation (scaled to 16-bit)')
        
        # Draw a box showing the playable area in the worldmap
        # The playable area is the central 1024x1024 region of the worldmap (not the full worldmap)
        h, w = worldmap.shape
        center_h, center_w = h//2, w//2
        box_size = 1024
        half_box = box_size // 2
        rect = plt.Rectangle((center_w - half_box, center_h - half_box), 
                             box_size, box_size, 
                             linewidth=2, edgecolor='r', facecolor='none')
        ax2.add_patch(rect)
        ax2.text(center_w, center_h - half_box - 30, "Playable Area", 
                 color='r', ha='center', va='bottom', fontsize=12)
        
        # Also draw a box in the base heightmap to highlight that it corresponds
        # to a larger area than just the playable region
        # According to Heightmaps.md, the base heightmap (4096x4096) covers a larger area 
        # than the playable area (which is 1024x1024 pixels in the worldmap)
        # The playable area is actually 14336x14336 meters, which is ~44% of the 32768x32768m heightmap
        playable_pixels = int(4096 * (14336 / 32768))  # Scale to find how many pixels represent the playable area
        h_base, w_base = base_heightmap.shape
        center_h_base, center_w_base = h_base//2, w_base//2
        half_playable = playable_pixels // 2
        rect_base = plt.Rectangle((center_w_base - half_playable, center_h_base - half_playable), 
                                 playable_pixels, playable_pixels, 
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax1.add_patch(rect_base)
        ax1.text(center_w_base, center_h_base - half_playable - 30, "Playable Area", 
                 color='r', ha='center', va='bottom', fontsize=12)
    else:
        ax2.text(0.5, 0.5, "Worldmap not generated", 
                 ha='center', va='center', fontsize=16)
        ax2.set_axis_off()
    
    plt.tight_layout()
    
    # Save the visualization
    filename_base = os.path.splitext(os.path.basename(heightmap_info['png']))[0]
    viz_path = os.path.join(output_dir, f"{filename_base}_visualization.png")
    plt.savefig(viz_path, dpi=300)
    plt.close()
    
    print(f"Visualization saved to {viz_path}")
    print("\nHeightmaps are ready for use in Cities: Skylines II")
    print("Remember: Each pixel represents 8x8 meters")
    print("- Base heightmap covers 32.768 x 32.768 km")
    print("- Worldmap shows a larger area with the central 8.192 x 8.192 km (1024x1024) matching the playable area")

# Replace the old visualize_heightmap function with the new one
visualize_heightmap = visualize_heightmaps

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate heightmaps for Cities Skylines 2')
    parser.add_argument('--lat', type=float, required=True, help='Center latitude')
    parser.add_argument('--lon', type=float, required=True, help='Center longitude')
    parser.add_argument('--output', type=str, default='elevation_output', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    parser.add_argument('--no-worldmap', action='store_true', help='Skip worldmap generation')
    parser.add_argument('--name', type=str, help='Custom name for output files (overrides auto-naming)')
    return parser.parse_args()

def main():
    """Main function to run the script."""
    args = parse_arguments()
    
    # Print intro
    print("="*80)
    print("Cities Skylines 2 Heightmap Generator")
    print("="*80)
    print(f"Generating heightmap for location: {args.lat}, {args.lon}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get location name for file naming
    location_name = None
    if args.name:
        location_name = args.name
        print(f"Using custom name: {location_name}")
    else:
        location_name = get_location_name(args.lat, args.lon)
        if location_name:
            print(f"Found location name: {location_name}")
        else:
            # If we can't get a location name, use coordinates
            location_name = f"lat{args.lat:.6f}_lon{args.lon:.6f}"
            print(f"Using coordinates as name: {location_name}")
    
    # Calculate bounding box
    bbox = calculate_bounding_box(args.lat, args.lon)
    print(f"Bounding box: {bbox}")
    
    # Query available elevation data
    elevation_data_info = query_available_elevation_data(bbox)
    
    # Download elevation data
    raw_tiff_path, source_type, resolution = download_elevation_data(elevation_data_info, bbox, args.output) 
    
    if raw_tiff_path:
        print(f"Successfully downloaded elevation data: {raw_tiff_path}")
        print(f"Source: {source_type}")
        if resolution:
            print(f"Resolution: {resolution}m")
        else:
            print("Resolution: Unknown (using direct service)")
            
        # Process heightmap
        heightmap_info = process_heightmap(raw_tiff_path, args.output, location_name, 
                                           create_worldmap=not args.no_worldmap)
        
        if heightmap_info:
            print(f"\nHeightmap processing complete!")
            print(f"- Elevation range: {heightmap_info['min_elev']:.2f}m to {heightmap_info['max_elev']:.2f}m")
            print(f"- Total relief: {heightmap_info['range']:.2f}m")
            print(f"\nFiles created:")
            print(f"- Base Heightmap PNG: {heightmap_info['png']}")
            print(f"- Base Heightmap TIFF: {heightmap_info['tiff']}")
            
            if heightmap_info['worldmap']:
                print(f"- Worldmap PNG: {heightmap_info['worldmap']['png']}")
                print(f"- Worldmap TIFF: {heightmap_info['worldmap']['tiff']}")
            
            # Visualize if requested
            if args.visualize:
                visualize_heightmaps(heightmap_info, args.lat, args.lon, args.output)
        else:
            print("Heightmap processing failed.")
    else:
        print("Failed to download elevation data for the specified location.")
        print("Possible reasons:")
        print("1. No elevation data available for this location")
        print("2. Server issues or rate limiting")
        print("3. Network connectivity problems")
        print("\nTry a different location or check your internet connection.")

if __name__ == "__main__":
    main()
