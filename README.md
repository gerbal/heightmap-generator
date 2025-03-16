# Cities Skylines II Heightmap and Worldmap Generator

This project is intended to output a Cities Skylines II Heightmap and Worldmap from a place name or a set of coordinates. 

We can use ArcGIS REST servers like `https://index.nationalmap.gov/arcgis/rest/services/3DEPElevationIndex/MapServer/` to get elevation data and output the required files. This project will fetch the DEM or other Heightmap data from an ArcGIS REST server, then convert the data to a height map (4096x4096 pixels, 14336 meters (14.336 km x 14.336 km)) and a world map (57344 meters) in Grayscale, 16-bit color channel depth, .png or .tiff.

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/githubnext/workspace-blank.git
    cd workspace-blank
    ```

2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage

1. Run the main script with a place name or coordinates as input:
    ```
    python main.py "Place Name" or
    python main.py "latitude,longitude"
    ```

2. The generated heightmap and world map will be saved in the output directory.

## Example

To generate a heightmap and world map for New York City:
```
python main.py "New York City"
```

To generate a heightmap and world map for specific coordinates:
```
python main.py "40.7128,-74.0060"
```

## Project Description

This project is intended to output a Cities Skylines II Heightmap and Worldmap from a place name or a set of coordinates. We can use ArcGIS REST servers like `https://index.nationalmap.gov/arcgis/rest/services/3DEPElevationIndex/MapServer/` to get elevation data and output the required files. This project will fetch the DEM or other Heightmap data from an ArcGIS REST server, then convert the data to a height map (4096x4096 pixels, 14336 meters (14.336 km x 14.336 km)) and a world map (57344 meters) in Grayscale, 16-bit color channel depth, .png or .tiff.

## Installation Instructions

1. Clone the repository:
    ```
    git clone https://github.com/githubnext/workspace-blank.git
    cd workspace-blank
    ```

2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

## Example Usage

To generate a heightmap and world map for New York City:
```
python main.py "New York City"
```

To generate a heightmap and world map for specific coordinates:
```
python main.py "40.7128,-74.0060"
```
