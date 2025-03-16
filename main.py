import sys
from fetch_elevation_data import fetch_elevation_data, parse_elevation_data
from convert_to_heightmap import convert_to_heightmap
from generate_world_map import generate_world_map

def main(input_data):
    try:
        # Fetch elevation data
        elevation_data = fetch_elevation_data(input_data)
        parsed_data = parse_elevation_data(elevation_data)

        # Convert to heightmap
        heightmap_image = convert_to_heightmap(parsed_data)

        # Generate world map
        world_map_image = generate_world_map(parsed_data)

        print("Heightmap and world map generated successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <place_name_or_coordinates>")
    else:
        input_data = sys.argv[1]
        main(input_data)
