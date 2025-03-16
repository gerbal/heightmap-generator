# Cities Skylines II Heightmap Generator

This tool generates heightmaps compatible with Cities Skylines 2 using high-resolution elevation data from USGS 3DEP (3D Elevation Program). It creates both the base heightmap and extended worldmap as required by the game. Currently limited to locations within the continental United States to ensure highest quality data.

## Features

- Generate heightmaps from any location in the continental US using coordinates or place names
- Automatic download of highest resolution available USGS 3DEP data (up to 1-meter resolution where available)
- Creates 16-bit PNG and TIFF heightmaps at optimal resolution
- Generates both detailed core heightmap (14.336 x 14.336 km) and extended worldmap (57.344 x 57.344 km)
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
3. Visualization image showing both maps with area indicators

## Technical Details

- **Core heightmap**: 4096×4096 pixels covering 14.336×14.336 km (3.5m per pixel)
- **Game playable area**: Matches the core heightmap area
- **Extended worldmap**: 4096×4096 pixels covering 57.344×57.344 km
- **Format**: 16-bit grayscale PNG/TIFF
- **Data source**: USGS 3DEP (3D Elevation Program)
- **Resolution**: Up to 1-meter where available, minimum 10-meter nationwide

## Requirements

- Python 3.7+
- Internet connection for downloading USGS elevation data
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

## Importing into Cities Skylines 2

1. Place the generated `.png` files in your game's heightmap folder:
   - Core heightmap: `<location_name>.png`
   - Worldmap: `<location_name>_worldmap.png`
   
   The default location is: `C:\Users\<username>\AppData\LocalLow\Colossal Order\Cities Skylines II\Maps`

2. Start a new game and select "Create New Map"
3. Choose your heightmap from the list

## Limitations

- Currently only supports locations within the continental United States
- Elevation data quality varies by region (best in urban and high-interest areas)
- Maximum resolution depends on available USGS 3DEP data for the area
