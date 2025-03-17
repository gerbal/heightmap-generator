# Cities Skylines II Heightmap Generator

This tool generates heightmaps compatible with Cities Skylines 2 using high-resolution elevation data from USGS 3DEP (3D Elevation Program). It creates both the base heightmap and extended worldmap as required by the game. Currently limited to locations within the continental United States to ensure highest quality data.

## Features

- Generate heightmaps from any location in the continental US using coordinates or place names
- Automatic download of highest resolution available USGS 3DEP data (up to 1-meter resolution where available)
- Creates 16-bit PNG and TIFF heightmaps at optimal resolution
- Generates both detailed core heightmap (14.336 x 14.336 km) and extended worldmap (57.344 x 57.344 km)
- **NEW:** Optional topobathymetry data integration for realistic underwater terrain
- Ensures consistent elevation scaling between core area and worldmap
- Handles no-data values and seamlessly fills gaps
- Includes visualization of the generated heightmaps with area indicators

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/heightmap-generator.git
   cd heightmap-generator
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Using main.py (Recommended)

The simplified interface accepts either place names or coordinates:

```bash
python main.py "Mount Mitchell, NC"
```

or

```bash
python main.py "35.7648,-82.2652"
```

### With Bathymetry Data (for coastal or lake areas)

To add realistic underwater terrain for water bodies:

```bash
python main.py "New York" --with-bathymetry
```

You can also specify the bathymetry data source:

```bash
python main.py "San Francisco" --with-bathymetry --bathymetry-source gebco
```

Available bathymetry sources:
- `noaa_dem` (default) - NOAA Digital Elevation Model (highest resolution)
- `noaa_coastal` - NOAA Coastal Relief Model
- `noaa_ninth_arc` - NOAA 9th Arc-Second Topobathy (3m resolution, limited coverage)
- `gebco` - Global Bathymetry data (global coverage, lower resolution)
- `noaa_crm` - NOAA Coastal Relief Model via ArcGIS

### Using heightmap_generator.py (Advanced)

For more control over the generation process:

```bash
python heightmap_generator.py --lat 35.7648 --lon -82.2652 --visualize
```

#### Required arguments:
- `--lat` - Latitude of the center point
- `--lon` - Longitude of the center point

#### Optional arguments:
- `--output` - Output directory (default: 'elevation_output')
- `--visualize` - Create visualization of the heightmaps
- `--no-worldmap` - Skip worldmap generation
- `--name` - Custom name for output files (overrides automatic location naming)

## Output Files

The generator creates:
1. Core heightmap (4096x4096) covering 14.336 x 14.336 km playable area
   - 16-bit PNG (game format)
   - 16-bit TIFF (for GIS applications)
2. Extended worldmap (4096x4096) covering 57.344 x 57.344 km area
   - 16-bit PNG (game format)
   - 16-bit TIFF (for GIS applications)
3. When using bathymetry:
   - Additional heightmaps with topobathymetry integration
   - Bathymetry visualization showing underwater terrain details
4. Visualization image showing all maps with area indicators

## Technical Details

- **Core heightmap**: 4096×4096 pixels covering 14.336×14.336 km (3.5m per pixel)
- **Game playable area**: Matches the core heightmap area
- **Extended worldmap**: 4096×4096 pixels covering 57.344×57.344 km
- **Format**: 16-bit grayscale PNG/TIFF
- **Data sources**: 
  - Topography: USGS 3DEP (3D Elevation Program)
  - Bathymetry: NOAA Digital Elevation Models, GEBCO Global Bathymetry
- **Resolution**: Up to 1-meter where available, minimum 10-meter nationwide

## Requirements

- Python 3.7+
- Internet connection for downloading elevation and bathymetry data
- Dependencies:
  - requests
  - numpy
  - Pillow
  - matplotlib
  - rasterio
  - scipy
  - geopy (for place name geocoding)

## Examples

1. Generate a heightmap for Great Smoky Mountains:
   ```bash
   python main.py "Great Smoky Mountains National Park"
   ```

2. Generate a heightmap for Mount Rainier:
   ```bash
   python main.py "Mount Rainier"
   ```

3. Generate a heightmap for a specific location in the Grand Canyon:
   ```bash
   python main.py "36.0544,-112.1401"
   ```

4. Generate a heightmap with bathymetry for coastal cities:
   ```bash
   python main.py "Seattle" --with-bathymetry
   python main.py "New York" --with-bathymetry
   python main.py "San Francisco" --with-bathymetry --bathymetry-source gebco
   ```

## Importing into Cities Skylines 2

1. Place the generated `.png` files in your game's heightmap folder:
   - Core heightmap: `<location_name>.png`
   - Worldmap: `<location_name>_worldmap.png`
   - With bathymetry: `<location_name>_topo_<source>.png`
   
   The default location is: `C:\Users\<username>\AppData\LocalLow\Colossal Order\Cities Skylines II\Maps`

2. Start a new game and select "Create New Map"
3. Choose your heightmap from the list

## Limitations

- Currently only supports locations within the continental United States
- Elevation data quality varies by region (best in urban and high-interest areas)
- Maximum resolution depends on available USGS 3DEP data for the area
- Bathymetry data coverage and quality varies, with best results in coastal areas and major water bodies

## Troubleshooting

### Data Download Issues

1. **Corrupted TIFF Files**
   - If you see errors like `TIFFReadEncodedTile() failed` or `Read failed`, the downloaded TIFF file may be corrupted
   - This can happen due to:
     - Network interruptions during download
     - Server timeout issues
     - Insufficient disk space
   - Solutions:
     - Try running the script again (it will download fresh data)
     - Check your internet connection
     - Ensure you have enough free disk space (at least 500MB recommended)

2. **USGS 3DEP or NOAA Service Issues**
   - The elevation services may occasionally be unavailable or slow
   - Common errors:
     - HTTP timeout errors
     - Invalid JSON responses
     - Incomplete data downloads
   - Solutions:
     - Wait a few minutes and try again
     - Try during off-peak hours (US daytime can be busier)
     - Check [USGS Status Page](https://status.usgs.gov) or [NOAA Status](https://status.noaa.gov) for service updates

### Processing Issues

1. **Memory Errors**
   - Processing large elevation datasets requires significant RAM
   - Recommended: 8GB RAM minimum, 16GB+ for optimal performance
   - If you see memory errors:
     - Close other memory-intensive applications
     - Try a smaller area first
     - Consider upgrading your RAM if issues persist

2. **Resolution and File Size**
   - High-resolution data (1m) can result in very large files
   - Each heightmap is 4096x4096 pixels (32MB for 16-bit)
   - Worldmaps cover 4x larger area
   - If file sizes are an issue:
     - Ensure sufficient disk space
     - Consider using lower resolution data
     - Clean up temporary files regularly

### Bathymetry Issues

1. **No Bathymetry Data Available**
   - Some inland areas may have limited or no bathymetry data
   - NOAA data primarily covers coastal areas and major lakes
   - If bathymetry fails, the tool will continue without it
   - Try using a different bathymetry source with `--bathymetry-source` option

2. **Unrealistic Underwater Terrain**
   - Bathymetry data quality and resolution varies significantly
   - Small water bodies may have simplified or synthetic depths
   - If underwater terrain looks unrealistic:
     - Try a different bathymetry source
     - Shift your location slightly closer to deeper water

### Known Limitations

1. **Maximum Height Handling**
   - CS2 has a maximum height limit of 4096 meters
   - Areas exceeding this will be clipped
   - Very high elevation ranges may lose detail in lower areas
   - Consider using custom scaling for extreme terrain

2. **No-Data Areas**
   - Some regions may have gaps in elevation data
   - The tool attempts to interpolate these areas
   - Large no-data regions may result in flat or unrealistic terrain
   - Try shifting the area slightly if this occurs

### Getting Help

If you encounter issues not covered here:
1. Check the full error message for specific details
2. Look for similar issues in the project's issue tracker
3. When reporting issues, include:
   - Full error message
   - Location coordinates
   - Operating system and Python version
   - Screenshots of any visual artifacts
