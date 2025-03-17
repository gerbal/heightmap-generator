"""
Bathymetry Data Fetcher Module

This module provides functions for fetching and processing bathymetric data
from various sources (NOAA, GEBCO, etc.) to enhance water features in heightmaps.
"""

import requests
import os
import json
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import hashlib
import matplotlib.pyplot as plt
from scipy import ndimage
import time
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('bathymetry_fetcher')

# Sources for bathymetric data
BATHYMETRY_SOURCES = {
    'noaa_coastal': {
        'name': 'NOAA Coastal Relief Model',
        'url': 'https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_global_mosaic/ImageServer/exportImage',
        'description': 'High-resolution coastal bathymetry for US coastal areas'
    },
    'noaa_ninth_arc': {
        'name': 'NOAA 9th Arc-Second Topo-Bathy',
        'url': 'https://coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/',
        'description': 'High-resolution (1/9 arc-second) topobathy for US coastal areas'
    },
    'gebco': {
        'name': 'GEBCO Global Bathymetry',
        'url': 'https://www.gebco.net/data_and_products/gebco_web_services/web_map_service/mapserv',
        'description': 'Global bathymetry grid at 15 arc-second intervals'
    },
    'noaa_crm': {
        'name': 'NOAA Coastal Relief Model (ArcGIS)',
        'url': 'https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_global_mosaic/ImageServer/exportImage',
        'description': 'NOAA Coastal Relief Model accessed via ArcGIS Image Service'
    },
    'noaa_dem': {
        'name': 'NOAA Digital Elevation Model (High-Res)',
        'url': 'https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_all/ImageServer/exportImage',
        'description': 'Highest resolution NOAA DEM available (includes both topo and bathy)'
    }
}

# Special regional datasets
REGIONAL_DATASETS = {}

def fetch_bathymetry_data(bbox, output_dir="elevation_output", sources=None, resolution=1024, cache=True):
    """
    Fetch bathymetric data from available sources for the given bounding box.
    
    Args:
        bbox (tuple): Bounding box as (min_lon, min_lat, max_lon, max_lat)
        output_dir (str): Directory to save the bathymetry data
        sources (list): List of source names to try (default: all available sources)
        resolution (int): Requested resolution (default: 1024x1024)
        cache (bool): Whether to use cached data if available
        
    Returns:
        tuple: (numpy.ndarray or None, dict or None) - Bathymetry data array and metadata if successful
    """
    logger.info(f"Fetching bathymetric data for bounding box: {bbox}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique filename for caching
    bbox_str = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_{resolution}_bathy"
    bbox_hash = hashlib.md5(bbox_str.encode()).hexdigest()
    cache_file_path = os.path.join(output_dir, f"bathymetry_{bbox_hash}.tiff")
    metadata_file_path = os.path.join(output_dir, f"bathymetry_{bbox_hash}_meta.json")
    
    # Check if the data is already cached
    if cache and os.path.exists(cache_file_path) and os.path.exists(metadata_file_path):
        logger.info(f"Using cached bathymetry data from {cache_file_path}")
        try:
            with open(metadata_file_path, 'r') as f:
                metadata = json.load(f)
            
            # If we want to force a specific source and the cached data is from a different source,
            # ignore the cache and fetch new data
            if sources and len(sources) == 1 and metadata.get('source') != sources[0]:
                logger.info(f"Cached data is from {metadata.get('source')} but we want {sources[0]}, fetching new data")
            else:
                with rasterio.open(cache_file_path) as src:
                    bathy_data = src.read(1)
                    rasterio_meta = src.meta
                    metadata['rasterio_meta'] = rasterio_meta
                    logger.info(f"Successfully loaded cached bathymetry data from {metadata.get('source_name')}")
                    return bathy_data, metadata
        except Exception as e:
            logger.warning(f"Error reading cached bathymetry data: {e}")
            # Continue to fetch new data if cached read fails
    

    # Default sources if none specified - prioritize NOAA DEM which has the most consistent results
    if sources is None:
        sources = ['noaa_dem', 'noaa_coastal', 'gebco', 'noaa_crm']
        logger.info(f"No sources specified, using defaults: {sources}")
    
    # Try each source until we get data
    for source_name in sources:
        if source_name not in BATHYMETRY_SOURCES:
            logger.warning(f"Unknown bathymetry source '{source_name}'")
            continue
        
        source = BATHYMETRY_SOURCES[source_name]
        logger.info(f"Trying bathymetry source: {source['name']}")
        
        try:
            # Call the appropriate fetch method based on source
            if source_name == 'noaa_coastal':
                logger.info("Fetching from NOAA Coastal Relief Model")
                result = fetch_from_noaa_coastal(bbox, resolution, cache_file_path)
            elif source_name == 'noaa_ninth_arc':
                logger.info("Fetching from NOAA 9th Arc-Second using default URL")
                result = fetch_from_noaa_ninth_arc(bbox, resolution, cache_file_path)
            elif source_name == 'gebco':
                logger.info("Fetching from GEBCO Global Bathymetry")
                result = fetch_from_gebco(bbox, resolution, cache_file_path)
            elif source_name == 'noaa_crm':
                logger.info("Fetching from NOAA Coastal Relief Model (ArcGIS)")
                result = fetch_from_noaa_crm(bbox, resolution, cache_file_path)
            elif source_name == 'noaa_dem':
                logger.info("Fetching from NOAA High-Resolution DEM")
                result = fetch_from_noaa_dem(bbox, resolution, cache_file_path)
            else:
                logger.warning(f"No fetch method implemented for source: {source_name}")
                result = None
            
            if result is not None:
                bathy_data, bathy_meta = result
                
                # Validate that the data contains valid information
                if np.all(np.isnan(bathy_data)):
                    logger.warning(f"All values in data from {source_name} are NaN, skipping")
                    continue
                
                # Check for too many zero values - may indicate data quality issue
                zero_percent = np.sum(bathy_data == 0) / bathy_data.size * 100
                if zero_percent > 95:
                    logger.warning(f"Data from {source_name} contains {zero_percent:.1f}% zero values, "
                                 f"which may indicate a data quality issue. Will try next source.")
                    continue
                
                # Save metadata to accompany the bathymetry data
                metadata = {
                    'source': source_name,
                    'source_name': source['name'],
                    'bbox': bbox,
                    'resolution': resolution,
                    'timestamp': time.time(),
                    'data_shape': bathy_data.shape,
                    'min_depth': float(np.nanmin(bathy_data)),
                    'max_depth': float(np.nanmax(bathy_data)),
                    'mean_depth': float(np.nanmean(bathy_data)),
                    'has_nan': bool(np.isnan(bathy_data).any()),
                    'zero_percent': float(zero_percent)
                }
                
                with open(metadata_file_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Successfully fetched bathymetry data from {source['name']}")
                metadata['rasterio_meta'] = bathy_meta
                return bathy_data, metadata
            else:
                logger.warning(f"No data returned from {source['name']}")
        
        except Exception as e:
            logger.error(f"Error fetching bathymetry data from {source['name']}: {e}", exc_info=True)
            continue
    
    logger.warning("No bathymetry data available from any source for this location")
    return None, None

def fetch_from_noaa_coastal(bbox, resolution, output_path):
    """
    Fetch bathymetric data from NOAA Coastal Relief Model.
    
    Args:
        bbox (tuple): Bounding box as (min_lon, min_lat, max_lon, max_lat)
        resolution (int): Requested resolution
        output_path (str): Path to save the resulting TIFF
        
    Returns:
        tuple: (numpy.ndarray, dict) - Bathymetry data array and metadata if successful
    """
    # Use NOAA NCEI direct GeoTIFF service instead of metadata endpoint
    url = "https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_global_mosaic/ImageServer/exportImage"
    
    # Calculate the best possible pixelSize based on bbox width and resolution
    lon_span = abs(bbox[2] - bbox[0])
    lat_span = abs(bbox[3] - bbox[1])
    pixel_size = min(lon_span / resolution, lat_span / resolution)
    
    params = {
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "bboxSR": 4326,  # WGS84
        "size": f"{resolution},{resolution}",
        "imageSR": 4326,
        "format": "tiff",
        "pixelType": "F32",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image"  # Request direct image download instead of JSON
    }
    
    # Optionally use pixelSize parameter instead of size for higher precision
    # when the area is not square
    if lon_span / lat_span > 1.2 or lat_span / lon_span > 1.2:
        # For non-square areas, use pixelSize instead of size for better resolution
        del params["size"]
        params["pixelSize"] = f"{pixel_size},{pixel_size}"
        logger.info(f"Using pixel size of {pixel_size} degrees for non-square area")
    
    logger.info(f"Sending request to NOAA Coastal Relief Model API with params: {params}")
    try:
        # Get the image data directly
        response = requests.get(url, params=params, stream=True)
        if response.status_code == 200:
            # Check content type to verify we got the expected format
            content_type = response.headers.get('content-type', '')
            if 'tiff' in content_type or 'image' in content_type:
                # Save the raw bathymetry data
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                # Load and return the bathymetry data
                try:
                    with rasterio.open(output_path) as src:
                        bathy_data = src.read(1)
                        bathy_meta = src.meta
                        logger.info(f"NOAA bathymetry data shape: {bathy_data.shape}")
                        return bathy_data, bathy_meta
                except rasterio.RasterioIOError as e:
                    logger.error(f"Error reading bathymetry data: {e}")
                    return None
            else:
                logger.error(f"Unexpected content type: {content_type}")
                return None
        else:
            logger.error(f"Failed to download bathymetry data, status code: {response.status_code}")
            logger.error(f"Response content: {response.content[:200]}...")
            return None
    except Exception as e:
        logger.error(f"Error fetching NOAA coastal bathymetry: {e}")
        return None
    logger.warning("No NOAA coastal bathymetry data available for this location")
    return None

def fetch_from_noaa_ninth_arc(bbox, resolution, output_path, region_url=None):
    """
    Fetch high-resolution bathymetric data from NOAA 1/9 arc-second topobathy dataset.
    This is a specialized function with higher resolution data (1/9 arc-second ≈ 3m).
    
    Args:
        bbox (tuple): Bounding box as (min_lon, min_lat, max_lon, max_lat)
        resolution (int): Requested resolution
        output_path (str): Path to save the resulting TIFF
        region_url (str): Optional specific URL for regional data
        
    Returns:
        tuple: (numpy.ndarray, dict) - Bathymetry data array and metadata if successful
    """
    # Default URL for NOAA ninth arc-second data
    base_url = region_url if region_url else BATHYMETRY_SOURCES['noaa_ninth_arc']['url']
    logger.info(f"Using base URL for NOAA 9th Arc-Second: {base_url}")
    
    try:
        # For New York Harbor area
        if bbox[0] >= -75.0 and bbox[0] <= -73.5 and bbox[1] >= 40.0 and bbox[1] <= 41.0:
            # This is a direct URL to the NY Harbor raster
            # Try different possible file paths for the NY Harbor data
            possible_paths = [
                f"{base_url}/nyc_harbor_13_navd88_2014.tif",  # Direct in the region_url
                f"{base_url}/nyc_harbor/nyc_harbor_13_navd88_2014.tif",  # In a subdirectory
                f"{base_url}/nyc_harbor/ny_harbor_13_navd88_2014.tif",  # Alternative filename
                f"{base_url}/nyc/nyc_harbor_13_navd88_2014.tif"  # Another subdirectory structure
            ]
            
            # Try each possible URL
            for download_url in possible_paths:
                logger.info(f"Trying NY Harbor topobathy URL: {download_url}")
                response = requests.head(download_url)
                
                if response.status_code == 200:
                    # Found a working URL, now download the data
                    logger.info(f"Found valid NY Harbor topobathy URL: {download_url}")
                    response = requests.get(download_url, stream=True)
                    
                    if response.status_code == 200:
                        # Save the raw data
                        logger.info(f"Downloading NY Harbor topobathy data from {download_url}")
                        with open(output_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        logger.info(f"Downloaded NY Harbor topobathy data to {output_path}")
                        
                        # Crop to our specific bbox
                        with rasterio.open(output_path) as src:
                            logger.info(f"Opened NY Harbor tiff, shape: {src.shape}, bounds: {src.bounds}")
                            
                            # Transform the bbox to pixel coordinates
                            # NY Harbor raster covers approximately -74.3 to -73.7, 40.4 to 41.0
                            # We need to crop just our area of interest
                            
                            # Get the window corresponding to our bbox
                            from rasterio.windows import from_bounds
                            try:
                                window = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], src.transform)
                                logger.info(f"Window from bbox: {window}")
                                
                                # Read the data for this window
                                data = src.read(1, window=window)
                                logger.info(f"Read window data, shape: {data.shape}")
                                
                                # Get the transform for the window
                                window_transform = src.window_transform(window)
                                
                                # Create a new metadata dict for the cropped data
                                meta = src.meta.copy()
                                meta.update({
                                    'height': data.shape[0],
                                    'width': data.shape[1],
                                    'transform': window_transform
                                })
                                
                                # If we need to resample to the requested resolution
                                if data.shape[0] != resolution or data.shape[1] != resolution:
                                    logger.info(f"Resampling from {data.shape} to {resolution}x{resolution}")
                                    # Simple resize using scipy's zoom
                                    from scipy.ndimage import zoom
                                    zoom_factor = (resolution/data.shape[0], resolution/data.shape[1])
                                    data = zoom(data, zoom_factor, order=1)
                                    
                                    # Update metadata for the resampled data
                                    meta.update({
                                        'height': resolution,
                                        'width': resolution,
                                        'transform': rasterio.transform.from_bounds(
                                            bbox[0], bbox[1], bbox[2], bbox[3], resolution, resolution)
                                    })
                                
                                # Save the cropped and resampled data
                                with rasterio.open(output_path, 'w', **meta) as dst:
                                    dst.write(data, 1)
                                logger.info(f"Saved cropped and resampled data to {output_path}")
                                
                                # Read the final data
                                with rasterio.open(output_path) as src2:
                                    bathy_data = src2.read(1)
                                    bathy_meta = src2.meta
                                
                                logger.info(f"NOAA ninth arc-second data shape: {bathy_data.shape}")
                                if np.isnan(bathy_data).all() or np.isinf(bathy_data).all():
                                    logger.warning("All values in bathymetry data are NaN or Inf")
                                    return None
                                
                                return bathy_data, bathy_meta
                            except Exception as e:
                                logger.error(f"Error processing window: {e}", exc_info=True)
                    else:
                        logger.warning(f"Failed to download from {download_url}, status code: {response.status_code}")
                else:
                    logger.warning(f"URL {download_url} not found, status code: {response.status_code}")
            

        # For other regions we could add similar specialized handlers
                
        logger.warning("No NOAA ninth arc-second topobathy data available for this location")
        return None
    
    except Exception as e:
        logger.error(f"Error processing NOAA ninth arc-second data: {e}", exc_info=True)
        return None

def fetch_from_gebco(bbox, resolution, output_path):
    """
    Fetch bathymetric data from GEBCO global bathymetry.
    
    Args:
        bbox (tuple): Bounding box as (min_lon, min_lat, max_lon, max_lat)
        resolution (int): Requested resolution
        output_path (str): Path to save the resulting TIFF
        
    Returns:
        tuple: (numpy.ndarray, dict) - Bathymetry data array and metadata if successful
    """
    # GEBCO WMS service
    url = "https://www.gebco.net/data_and_products/gebco_web_services/web_map_service/mapserv"
    
    # For GEBCO, try to request a higher resolution than the default
    # GEBCO has a resolution limit, but we'll request more to get the best available
    target_resolution = min(4000, resolution * 2)  # Double the resolution but cap at 4000
    
    params = {
        "request": "GetMap",
        "service": "WMS",
        "version": "1.1.1",
        "layers": "GEBCO_LATEST",
        "styles": "",
        "format": "image/tiff",
        "transparent": "true",
        "srs": "EPSG:4326",
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "width": target_resolution,
        "height": target_resolution
    }
    
    logger.info(f"Sending request to GEBCO API with params: {params}")
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            # Save the raw data
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            # Process GEBCO data - might need special handling
            with rasterio.open(output_path) as src:
                bathy_data = src.read(1)
                bathy_meta = src.meta
                
                # GEBCO data might need to be flipped or transformed
                # Bathymetry values are typically negative for below sea level
                # Make sure values below sea level are negative
                if np.nanmean(bathy_data) > 0 and np.nanmin(bathy_data) >= 0:
                    # If all values are positive but should represent depths
                    # we need to invert them
                    logger.info("GEBCO data has positive values for depths, inverting")
                    sea_mask = bathy_data < 0
                    bathy_data = -bathy_data
                    # Keep land (originally positive values) as positive
                    bathy_data[sea_mask] = -bathy_data[sea_mask]
                
                logger.info(f"GEBCO bathymetry data shape: {bathy_data.shape}")
                logger.info(f"GEBCO data range: {np.nanmin(bathy_data):.2f} to {np.nanmax(bathy_data):.2f}, mean: {np.nanmean(bathy_data):.2f}")
                return bathy_data, bathy_meta
        else:
            logger.error(f"GEBCO API returned status code {response.status_code}")
            logger.error(f"Response content: {response.text[:200]}...")
    except Exception as e:
        logger.error(f"Error processing GEBCO data: {e}")
    
    logger.warning("No GEBCO bathymetry data available for this location")
    return None

def fetch_from_noaa_crm(bbox, resolution, output_path):
    """
    Fetch bathymetric data from NOAA Coastal Relief Model via ArcGIS Image Service.
    
    Args:
        bbox (tuple): Bounding box as (min_lon, min_lat, max_lon, max_lat)
        resolution (int): Requested resolution
        output_path (str): Path to save the resulting TIFF
        
    Returns:
        tuple: (numpy.ndarray, dict) - Bathymetry data array and metadata if successful
    """
    url = BATHYMETRY_SOURCES['noaa_crm']['url']
    
    # Use simplified parameters based on successful testing
    # Calculate appropriate size to maintain aspect ratio
    lon_span = abs(bbox[2] - bbox[0])
    lat_span = abs(bbox[3] - bbox[1])
    aspect_ratio = lon_span / lat_span if lat_span != 0 else 1
    
    # Determine width and height based on aspect ratio
    if lon_span > lat_span:
        width = resolution
        height = int(resolution * (lat_span / lon_span))
    else:
        height = resolution
        width = int(resolution * (lon_span / lat_span))
    
    params = {
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "bboxSR": 4326,
        "size": f"{width},{height}",
        "format": "tiff",
        "pixelType": "F32",
        "f": "image"
    }
    
    logger.info(f"Sending request to NOAA CRM ArcGIS Image Service with params: {params}")
    
    try:
        response = requests.get(url, params=params, stream=True)
        
        if response.status_code == 200:
            # Save the raw data
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Load and return the bathymetry data
            with rasterio.open(output_path) as src:
                bathy_data = src.read(1)
                bathy_meta = src.meta
                logger.info(f"NOAA CRM data shape: {bathy_data.shape}")
                logger.info(f"NOAA CRM data range: min={np.nanmin(bathy_data):.2f}, max={np.nanmax(bathy_data):.2f}, mean={np.nanmean(bathy_data):.2f}")
                
                # Check for all zeros
                zero_percent = np.sum(bathy_data == 0) / bathy_data.size * 100
                if zero_percent > 95:
                    logger.warning(f"Data appears to be mostly zeros ({zero_percent:.1f}%), potential data issue")
                
                # For bathymetry, negative values are depths below sea level
                # If there's a mix of positive and negative values, it represents both land and water
                # We'll leave the values as is since we want both bathymetry and topography
                
                return bathy_data, bathy_meta
        else:
            logger.error(f"NOAA CRM ArcGIS Image Service returned status code {response.status_code}")
            logger.error(f"Response content: {response.content[:200]}...")
            return None
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during request to NOAA CRM ArcGIS Image Service: {e}")
        return None
    except rasterio.RasterioIOError as e:
        logger.error(f"Error reading raster data from downloaded file: {e}")
        return None
    
    logger.warning("No NOAA CRM bathymetry data available for this location")
    return None

def fetch_from_noaa_dem(bbox, resolution, output_path):
    """
    Fetch high-resolution bathymetric data from NOAA Digital Elevation Model service.
    This is the highest resolution NOAA DEM available and includes both topography and bathymetry.
    
    Args:
        bbox (tuple): Bounding box as (min_lon, min_lat, max_lon, max_lat)
        resolution (int): Requested resolution
        output_path (str): Path to save the resulting TIFF
        
    Returns:
        tuple: (numpy.ndarray, dict) - Bathymetry data array and metadata if successful
    """
    url = BATHYMETRY_SOURCES['noaa_dem']['url']
    
    # Use the simplest parameters that work with the NOAA DEM API
    params = {
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "bboxSR": 4326,
        "format": "tiff",
        "pixelType": "F32",
        "f": "image",
    }
    
    # Add size parameter based on resolution 
    # Check if the aspect ratio is close to 1:1
    lon_span = abs(bbox[2] - bbox[0])
    lat_span = abs(bbox[3] - bbox[1])
    
    aspect_ratio = lon_span / lat_span if lat_span != 0 else 1
    
    # For square-ish areas, use the size parameter
    if 0.8 <= aspect_ratio <= 1.2:
        params["size"] = f"{resolution},{resolution}"
        logger.info(f"Using size parameter: {resolution}x{resolution} for square-ish area")
    else:
        # For rectangular areas, calculate width and height to maintain aspect ratio
        if lon_span > lat_span:
            width = resolution
            height = int(resolution * (lat_span / lon_span))
        else:
            height = resolution
            width = int(resolution * (lon_span / lat_span))
        params["size"] = f"{width},{height}"
        logger.info(f"Using size parameter: {width}x{height} for rectangular area with aspect ratio {aspect_ratio:.2f}")
    
    logger.info(f"Sending request to NOAA High-Res DEM service with params: {params}")
    
    try:
        response = requests.get(url, params=params, stream=True)
        
        if response.status_code == 200:
            # Save the raw data
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Load and return the bathymetry data
            with rasterio.open(output_path) as src:
                bathy_data = src.read(1)
                bathy_meta = src.meta
                logger.info(f"NOAA High-Res DEM data shape: {bathy_data.shape}")
                
                # Convert positive depths to negative for consistency with other sources
                # NOAA DEM may have positive values for depths below sea level
                if np.nanmedian(bathy_data) > 0 and np.nanmin(bathy_data) < 0:
                    # If we have both positive and negative values, assume negative is land
                    # and positive is water (need to invert water values)
                    water_mask = bathy_data > 0
                    if np.any(water_mask):
                        logger.info("Converting positive depth values to negative")
                        bathy_data[water_mask] = -bathy_data[water_mask]
                
                return bathy_data, bathy_meta
        else:
            logger.error(f"NOAA High-Res DEM service returned status code {response.status_code}")
            logger.error(f"Response content: {response.content[:200]}...")
            return None
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during request to NOAA High-Res DEM service: {e}")
        return None
    except rasterio.RasterioIOError as e:
        logger.error(f"Error reading raster data from downloaded file: {e}")
        return None
    
    logger.warning("No NOAA High-Res DEM data available for this location")
    return None

def interpolate_bathymetry(bathy_grid):
    """
    Interpolate missing values in bathymetry grid.
    
    Args:
        bathy_grid (numpy.ndarray): Grid with NaN values to be interpolated
        
    Returns:
        numpy.ndarray: Interpolated bathymetry grid
    """
    mask = np.isnan(bathy_grid)
    
    # If the grid is empty or mostly NaN, return as is
    if np.all(mask):
        return bathy_grid
    
    # Copy to avoid modifying the original during interpolation
    filled = bathy_grid.copy()
    
    # For small to medium-sized gaps, use radial basis function interpolation
    # which produces smoother, more natural-looking results than gaussian filter
    if np.sum(mask) < 0.5 * bathy_grid.size:  # If less than half the data is missing
        try:
            from scipy.interpolate import Rbf
            
            # Get coordinates of valid data points
            y_indices, x_indices = np.where(~mask)
            values = bathy_grid[~mask]
            
            # Create interpolator - multiquadric usually gives good results for topography
            if len(values) > 100:
                # For many points, sample to speed up computation
                sample_size = min(5000, len(values))
                sample_idx = np.random.choice(len(values), sample_size, replace=False)
                y_sample = y_indices[sample_idx]
                x_sample = x_indices[sample_idx]
                values_sample = values[sample_idx]
                
                rbf = Rbf(y_sample, x_sample, values_sample, function='multiquadric')
            else:
                rbf = Rbf(y_indices, x_indices, values, function='multiquadric')
            
            # Get coordinates of points to interpolate
            y_missing, x_missing = np.where(mask)
            
            # Interpolate in batches to avoid memory issues
            batch_size = 10000
            for i in range(0, len(y_missing), batch_size):
                batch_y = y_missing[i:i+batch_size]
                batch_x = x_missing[i:i+batch_size]
                filled[batch_y, batch_x] = rbf(batch_y, batch_x)
                
            logger.info("Used RBF interpolation for high-quality bathymetry filling")
            
        except (ImportError, MemoryError, ValueError) as e:
            # Fall back to gaussian filter method
            logger.warning(f"RBF interpolation failed, falling back to gaussian filter: {e}")
            temp = ndimage.gaussian_filter(np.where(~mask, filled, 0), sigma=2)
            weights = ndimage.gaussian_filter(~mask * 1.0, sigma=2)
            
            # Normalize by the weights to get the interpolated values
            with np.errstate(divide='ignore', invalid='ignore'):
                filled[mask] = temp[mask] / weights[mask]
    else:
        # For largely empty grids, use the simpler gaussian filter method
        logger.info("Using gaussian filter for interpolation of sparse bathymetry data")
        temp = ndimage.gaussian_filter(np.where(~mask, filled, 0), sigma=2)
        weights = ndimage.gaussian_filter(~mask * 1.0, sigma=2)
        
        # Normalize by the weights to get the interpolated values
        with np.errstate(divide='ignore', invalid='ignore'):
            filled[mask] = temp[mask] / weights[mask]
    
    # For any remaining NaNs (e.g., far from all data points), use the mean
    remaining_mask = np.isnan(filled)
    if np.any(remaining_mask):
        valid_mean = np.nanmean(filled)
        filled[remaining_mask] = valid_mean
    
    return filled

def apply_bathymetry_to_heightmap(elevation_data, water_mask, bathymetry_data, bathy_meta,
                                  min_elev, max_depth=-30, coastal_depth=-1):
    """
    Apply bathymetry data to water areas in a heightmap.
    
    Args:
        elevation_data (numpy.ndarray): The elevation data to modify
        water_mask (numpy.ndarray): Boolean mask of water areas
        bathymetry_data (numpy.ndarray): Bathymetry data
        bathy_meta (dict): Metadata for the bathymetry data
        min_elev (float): Minimum elevation in the original heightmap
        max_depth (float): Maximum water depth in meters (negative)
        coastal_depth (float): Depth at coastlines in meters (negative)
        
    Returns:
        numpy.ndarray: Modified elevation data with bathymetry applied
    """
    if bathymetry_data is None or not np.any(water_mask):
        return elevation_data
    
    # Create a copy of the original data
    modified_elevation = elevation_data.copy()
    
    # Resize bathymetry data to match the elevation data if needed
    if (bathymetry_data.shape != elevation_data.shape) and ('rasterio_meta' in bathy_meta):
        print(f"Resampling bathymetry data from {bathymetry_data.shape} to {elevation_data.shape}")
        
        # Use rasterio to properly reproject and resample
        resampled_bathy = np.zeros_like(elevation_data, dtype=np.float32)
        
        # Get the CRS and bounds information from metadata
        src_crs = bathy_meta['rasterio_meta'].get('crs', 'EPSG:4326')
        src_transform = bathy_meta['rasterio_meta'].get('transform')
        
        # Calculate the transform for resampling
        dst_shape = elevation_data.shape
        dst_bounds = bathy_meta['bbox']
        
        dst_transform = rasterio.transform.from_bounds(
            dst_bounds[0], dst_bounds[1], dst_bounds[2], dst_bounds[3],
            dst_shape[1], dst_shape[0]
        )
        
        # Reproject the bathymetry data
        reproject(
            source=bathymetry_data,
            destination=resampled_bathy,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs='EPSG:4326',
            resampling=Resampling.bilinear
        )
        
        bathymetry_data = resampled_bathy
    else:
        # Fallback to simple resize
        from scipy.ndimage import zoom
        zoom_factor = (elevation_data.shape[0] / bathymetry_data.shape[0],
                      elevation_data.shape[1] / bathymetry_data.shape[1])
        bathymetry_data = zoom(bathymetry_data, zoom_factor, order=1)
    
    # Create a mask for valid bathymetry data (not NaN)
    valid_bathy_mask = ~np.isnan(bathymetry_data)
    
    # Create a combined mask for water areas with valid bathymetry
    combined_mask = water_mask & valid_bathy_mask
    
    if np.any(combined_mask):
        print(f"Applying bathymetry data to {np.sum(combined_mask)} water pixels")
        
        # Clip bathymetry depths to reasonable range (between coastal_depth and max_depth)
        clipped_bathy = np.clip(bathymetry_data, max_depth, coastal_depth)
        
        # Set elevations from bathymetry where we have valid data
        modified_elevation[combined_mask] = min_elev + clipped_bathy[combined_mask]
        
        # For water areas without valid bathymetry data, generate synthetic depths
        remaining_water = water_mask & ~valid_bathy_mask
        if np.any(remaining_water):
            print(f"Generating synthetic depths for {np.sum(remaining_water)} additional water pixels")
            
            # Calculate distance from shore for each water pixel
            distance = ndimage.distance_transform_edt(water_mask)
            max_distance = np.max(distance)
            
            if max_distance > 0:
                # Scale distances to get depths between coastal_depth and max_depth
                normalized_distance = distance / max_distance
                water_depths = coastal_depth + (max_depth - coastal_depth) * normalized_distance
                modified_elevation[remaining_water] = min_elev + water_depths[remaining_water]
            else:
                # Fallback: constant depth
                modified_elevation[remaining_water] = min_elev + coastal_depth
    
    return modified_elevation

def visualize_bathymetry(bathymetry_data, metadata, output_dir="elevation_output"):
    """
    Create a visualization of the bathymetry data.
    
    Args:
        bathymetry_data (numpy.ndarray): The bathymetry data
        metadata (dict): Metadata for the bathymetry data
        output_dir (str): Directory to save the visualization
        
    Returns:
        str: Path to the visualization file
    """
    if bathymetry_data is None:
        return None
    
    # Create a unique filename
    source_name = metadata.get('source_name', 'unknown')
    bbox = metadata.get('bbox', [0, 0, 0, 0])
    bbox_str = f"{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}"
    viz_path = os.path.join(output_dir, f"bathymetry_{source_name}_{bbox_str}.png")
    
    # Create the visualization
    plt.figure(figsize=(12, 10))
    
    # Use a better colormap for bathymetry - 'cividis' is perceptually uniform
    # and works well for depth visualization
    cmap = plt.cm.get_cmap('cividis').copy()
    
    # Create a masked array to handle NaN values
    masked_data = np.ma.masked_invalid(bathymetry_data)
    
    # Check if we have any depth variation
    min_depth = np.nanmin(bathymetry_data)
    max_depth = np.nanmax(bathymetry_data)
    mean_depth = np.nanmean(bathymetry_data)
    
    # If all depths are zero, add warning to the plot
    if min_depth == 0 and max_depth == 0:
        logger.warning("All bathymetry values are zero. Data quality issue detected.")
        # Add a small offset to max depth to avoid colormap issues
        max_depth = 0.1
    
    # Show the bathymetry data with a better visualization
    img = plt.imshow(masked_data, cmap=cmap, vmin=min_depth, vmax=max_depth)
    cbar = plt.colorbar(img, label='Depth (m)')
    
    # Add depth contours to highlight features - only if we have varying depths
    if min_depth != max_depth:
        contour_levels = [-100, -50, -20, -10, -5, -2, -1, 0]
        # Filter levels to only include those in our data range
        valid_levels = [level for level in contour_levels if min_depth <= level <= max_depth]
        if valid_levels:
            contours = plt.contour(masked_data, levels=valid_levels, colors='white', alpha=0.4)
            plt.clabel(contours, inline=True, fontsize=8, fmt='%d m')
    
    # Add metadata to the title
    resolution = bathymetry_data.shape
    title = f"Bathymetry Data: {source_name}\n"
    title += f"Depth Range: {min_depth:.1f}m to {max_depth:.1f}m (avg: {mean_depth:.1f}m)\n"
    title += f"Resolution: {resolution[0]}x{resolution[1]} pixels"
    
    if min_depth == 0 and max_depth == 0:
        title += "\nWARNING: All depths are zero. Possible data quality issue."
    
    plt.title(title)
    
    # Add spatial reference (fix the syntax error with double colon)
    plt.xlabel(f"Longitude: {bbox[0]:.4f} to {bbox[2]:.4f}")
    plt.ylabel(f"Latitude: {bbox[1]:.4f} to {bbox[3]:.4f}")
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig(viz_path, dpi=300)
    plt.close()
    
    print(f"Bathymetry visualization saved to {viz_path}")
    return viz_path