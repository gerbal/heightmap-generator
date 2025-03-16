import requests

def fetch_elevation_data(place_name_or_coordinates):
    url = "https://index.nationalmap.gov/arcgis/rest/services/3DEPElevationIndex/MapServer/export"
    params = {
        "bbox": place_name_or_coordinates,
        "bboxSR": 4326,
        "imageSR": 4326,
        "size": "4096,4096",
        "format": "json",
        "f": "json"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        raise Exception(f"Error fetching elevation data: {response.status_code}")

def parse_elevation_data(data):
    # Assuming the data contains a 'value' field with elevation values
    elevation_values = data['value']
    return elevation_values
