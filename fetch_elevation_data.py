import requests
import json
import numpy as np
from geopy.geocoders import Nominatim
import time
import os
import tempfile
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import zoom

# Continental US bounding box (approximate)
# Format: (min_lon, min_lat, max_lon, max_lat)
CONTINENTAL_US_BBOX = (-125.0, 24.0, -66.0, 49.5)

def is_in_continental_us(lat, lon):
    """
    Check if the given coordinates are within the continental United States.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        
    Returns:
        bool: True if coordinates are within continental US, False otherwise
    """
    min_lon, min_lat, max_lon, max_lat = CONTINENTAL_US_BBOX
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat

def geocode_place_name(place_name):
    """
    Convert a place name to coordinates using the Nominatim geocoding service.
    
    Args:
        place_name (str): The name of the place to geocode
        
    Returns:
        tuple: (latitude, longitude) or None if geocoding failed
    """
    try:
        geolocator = Nominatim(user_agent="HeightmapGenerator/1.0")
        location = geolocator.geocode(place_name)
        if location:
            return (location.latitude, location.longitude)
        return None
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None

def parse_coordinates(input_str):
    """
    Parse a string containing latitude and longitude.
    
    Args:
        input_str (str): String in format "latitude,longitude"
        
    Returns:
        tuple: (latitude, longitude) or None if parsing failed
    """
    try:
        parts = input_str.split(',')
        if len(parts) == 2:
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
            return (lat, lon)
        return None
    except:
        return None

def fetch_elevation_data(place_name_or_coordinates, for_worldmap=False):
    """
    Fetch elevation data for a location specified by name or coordinates.
    This function only works for locations within the continental United States.
    Uses USGS 3DEP services for high-resolution data within the US.
    
    Args:
        place_name_or_coordinates (str): Place name or "latitude,longitude" string
        for_worldmap (bool): If True, fetches data for the worldmap (larger area, can be lower resolution)
        
    Returns:
        dict: Elevation data and metadata
    """
    # Check if input is coordinates
    coordinates = parse_coordinates(place_name_or_coordinates)
    
    # If not coordinates, try geocoding the place name
    if not coordinates:
        print(f"Attempting to geocode '{place_name_or_coordinates}'...")
        coordinates = geocode_place_name(place_name_or_coordinates)
        
    if not coordinates:
        raise ValueError(f"Could not determine coordinates for '{place_name_or_coordinates}'")
        
    lat, lon = coordinates
    print(f"Using coordinates: {lat}, {lon}")
    
    # Check if coordinates are within continental US
    if not is_in_continental_us(lat, lon):
        raise ValueError(f"Coordinates ({lat}, {lon}) are outside the continental United States.")
    
    # Calculate appropriate bounding box based on request type
    from heightmap_generator import calculate_bounding_box
    if for_worldmap:
        # Worldmap covers 57.344 x 57.344 km
        bbox = calculate_bounding_box(lat, lon, width_km=57.344, height_km=57.344)
    else:
        # Core heightmap covers just the playable area (14.336 x 14.336 km)
        # This ensures we get maximum resolution where it matters most
        bbox = calculate_bounding_box(lat, lon, width_km=14.336, height_km=14.336)
    
    print(f"Using USGS 3DEP elevation service for {'worldmap' if for_worldmap else 'core heightmap'}...")
    
    # Query for available high-resolution data
    arcgis_url = "https://index.nationalmap.gov/arcgis/rest/services/3DEPElevationIndex/MapServer/find"
    params = {
        "searchText": "elevation",
        "contains": "true",
        "searchFields": "ProjectName",
        "sr": "4326",  # WGS84
        "layers": "all",
        "returnGeometry": "true",
        "f": "json",
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "geometry": json.dumps({
            "xmin": bbox[0],
            "ymin": bbox[1],
            "xmax": bbox[2],
            "ymax": bbox[3],
            "spatialReference": {"wkid": 4326}
        })
    }
    
    try:
        print("Checking for available high-resolution data...")
        response = requests.get(arcgis_url, params=params)
        data = response.json()
        
        if 'results' in data and len(data['results']) > 0:
            best_resolution = float('inf')
            best_result = None
            
            for result in data['results']:
                if 'attributes' in result:
                    attrs = result['attributes']
                    if 'Resolution' in attrs and attrs['Resolution'] < best_resolution:
                        best_resolution = attrs['Resolution']
                        best_result = result
            
            if best_result:
                print(f"Found best available resolution: {best_resolution}m")
    except Exception as e:
        print(f"Error querying resolution data: {e}")
    
    # Make the actual elevation request
    elevation_service_url = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
    
    # Start with 5120x5120 for core heightmap (still higher than base resolution but less likely to timeout)
    # For worldmap, keep the standard 4096x4096
    size = "4096,4096" if for_worldmap else "5120,5120"
    
    # Use cubic interpolation for best quality
    interpolation = "RSP_CubicConvolution" if not for_worldmap else "RSP_BilinearInterpolation"
    
    params = {
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "bboxSR": 4326,  # WGS84
        "size": size,
        "imageSR": 4326,
        "format": "tiff",
        "pixelType": "F32",
        "interpolation": interpolation,
        "f": "json"
    }
    
    try:
        print(f"Requesting elevation data at {size} resolution...")
        response = requests.get(elevation_service_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'href' in data:
            img_url = data['href']
            print(f"Found USGS 3DEP data: {img_url}")
            img_response = requests.get(img_url)
            img_response.raise_for_status()
            
            # Save the raw data to a temporary file
            temp_dir = tempfile.gettempdir()
            raw_tiff_path = os.path.join(temp_dir, f"raw_elevation{'_worldmap' if for_worldmap else ''}.tiff")
            
            with open(raw_tiff_path, 'wb') as f:
                f.write(img_response.content)
                
            print(f"Downloaded elevation data to {raw_tiff_path}")
            
            # Validate the downloaded data
            try:
                with rasterio.open(raw_tiff_path) as src:
                    print(f"Data resolution: {src.width}x{src.height}, {src.count} bands, dtype: {src.dtypes[0]}")
                    
                return {
                    'data_path': raw_tiff_path,
                    'lat': lat,
                    'lon': lon,
                    'bbox': bbox,
                    'source': 'USGS 3DEP',
                    'for_worldmap': for_worldmap
                }
            except Exception as e:
                print(f"Warning: TIFF validation failed: {e}")
                raise
    
    except Exception as e:
        print(f"Error fetching USGS elevation data: {e}")
        if not for_worldmap:  # Only try fallback for core heightmap
            print("Falling back to Open-Elevation API...")
            return fetch_from_open_elevation(lat, lon, bbox)
        raise

def fetch_from_open_elevation(lat, lon, bbox):
    """Fallback method using Open-Elevation API"""
    try:
        print("Using Open-Elevation API as fallback...")
        elevation_service_url = "https://api.open-elevation.com/api/v1/lookup"
        
        # Calculate points grid (200x200 points for better detail)
        lats = np.linspace(bbox[1], bbox[3], 200)
        lons = np.linspace(bbox[0], bbox[2], 200)
        points = [(lat, lon) for lat in lats for lon in lons]
        
        # Prepare request data
        locations = [{"latitude": lat, "longitude": lon} for lat, lon in points]
        request_data = {"locations": locations}
        
        # Get the elevation data
        response = requests.post(elevation_service_url, json=request_data)
        elevation_data = response.json()
        
        if 'results' not in elevation_data:
            raise Exception("Could not find elevation data for this location")
            
        # Convert results to numpy array and reshape to grid
        elevations = np.array([point['elevation'] for point in elevation_data['results']])
        elevation_grid = elevations.reshape(200, 200)
        
        # Interpolate to higher resolution (4096x4096)
        high_res_elevation = zoom(elevation_grid, 4096/200)
        
        # Save as GeoTIFF
        transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], 
                              high_res_elevation.shape[1], high_res_elevation.shape[0])
        
        # Save the raw data to a temporary file
        temp_dir = tempfile.gettempdir()
        raw_tiff_path = os.path.join(temp_dir, "raw_elevation.tiff")
        
        with rasterio.open(raw_tiff_path, 'w', 
                          driver='GTiff',
                          height=high_res_elevation.shape[0],
                          width=high_res_elevation.shape[1],
                          count=1,
                          dtype=high_res_elevation.dtype,
                          crs='+proj=latlong',
                          transform=transform) as dst:
            dst.write(high_res_elevation, 1)
        
        return {
            'data_path': raw_tiff_path,
            'lat': lat,
            'lon': lon,
            'bbox': bbox,
            'source': 'Open-Elevation API',
            'for_worldmap': False
        }
    
    except Exception as e:
        print(f"Error with Open-Elevation API: {e}")
        raise

def parse_elevation_data(data):
    """
    Parse the downloaded elevation data.
    
    Args:
        data (dict): Data from fetch_elevation_data()
        
    Returns:
        array: Numpy array of elevation values
    """
    try:
        import rasterio
        from scipy import ndimage
        
        with rasterio.open(data['data_path']) as src:
            elevation_array = src.read(1)
            
            # Handle no-data values
            no_data = src.nodata
            if no_data is not None:
                mask = elevation_array == no_data
                if mask.any():
                    from heightmap_generator import interpolate_nodata
                    elevation_array = interpolate_nodata(elevation_array, mask)
            
            # Handle NaN or infinite values
            if np.isnan(elevation_array).any() or np.isinf(elevation_array).any():
                mask = np.logical_or(np.isnan(elevation_array), np.isinf(elevation_array))
                from heightmap_generator import interpolate_nodata
                elevation_array = interpolate_nodata(elevation_array, mask)
        
        return elevation_array
    except Exception as e:
        print(f"Error parsing elevation data: {e}")
        raise
