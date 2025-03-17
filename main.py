#!/usr/bin/env python3
"""
Cities Skylines 2 Heightmap Generator - Simplified Interface
For Continental US locations only
"""

import sys
import os
import re
from fetch_elevation_data import fetch_elevation_data, parse_elevation_data, geocode_place_name, parse_coordinates, is_in_continental_us
import argparse
from heightmap_generator import process_heightmap, generate_worldmap, visualize_heightmaps

def main(input_data, output_dir="elevation_output", visualize=True):
    """
    Main function to process either place name or coordinates and generate heightmap.
    Limited to locations within the continental United States.
    
    Args:
        input_data (str): Place name or coordinates
        output_dir (str): Directory to save output files
        visualize (bool): Whether to create a visualization
    """
    try:
        print("="*80)
        print("Cities Skylines 2 Heightmap Generator (Continental US Only)")
        print("="*80)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # First determine if input is coordinates
        coordinates = parse_coordinates(input_data)
        
        if coordinates:
            lat, lon = coordinates
            # Verify coordinates are in continental US
            if not is_in_continental_us(lat, lon):
                print(f"Error: Coordinates ({lat}, {lon}) are outside the continental United States.")
                print("This tool is limited to continental US locations only.")
                return
            location_name = None
        else:
            # Try to geocode place name
            print(f"Looking up coordinates for '{input_data}'...")
            coordinates = geocode_place_name(input_data)
            if not coordinates:
                print(f"Could not find coordinates for '{input_data}'")
                return
            lat, lon = coordinates
            # Verify coordinates are in continental US
            if not is_in_continental_us(lat, lon):
                print(f"Error: '{input_data}' ({lat}, {lon}) is outside the continental United States.")
                print("This tool is limited to continental US locations only.")
                return
            location_name = input_data  # Use the provided place name
        
        print(f"Using location: {lat}, {lon}")
        
        # Fetch worldmap data first to get the full elevation range
        print("\nFetching data for extended worldmap (57.344 x 57.344 km area)...")
        worldmap_elevation_data = fetch_elevation_data(f"{lat},{lon}", for_worldmap=True)
        
        if not worldmap_elevation_data or 'data_path' not in worldmap_elevation_data:
            print("Failed to fetch worldmap elevation data. Cannot continue without elevation range.")
            return
            
        worldmap_raw_tiff = worldmap_elevation_data['data_path']
        print(f"Downloaded worldmap elevation data to {worldmap_raw_tiff}")
        
        # Process worldmap first to get elevation range
        worldmap_info = process_heightmap(
            worldmap_raw_tiff, 
            output_dir, 
            f"{location_name}_worldmap" if location_name else "worldmap"
        )
        
        if not worldmap_info:
            print("Failed to process worldmap. Cannot continue without elevation range.")
            return
        
        print(f"\nWorldmap elevation range: {worldmap_info['min_elev']:.2f}m to {worldmap_info['max_elev']:.2f}m")
        print(f"Total relief: {worldmap_info['range']:.2f}m")
        
        # Now fetch high-resolution data for core heightmap
        print("\nFetching high-resolution data for core playable area (14.336 x 14.336 km)...")
        core_elevation_data = fetch_elevation_data(f"{lat},{lon}", for_worldmap=False)
        
        if not core_elevation_data or 'data_path' not in core_elevation_data:
            print("Failed to fetch elevation data for core area")
            return
            
        raw_tiff_path = core_elevation_data['data_path']
        print(f"Downloaded core elevation data to {raw_tiff_path}")
        
        # Process the core heightmap using the worldmap's elevation range
        heightmap_info = process_heightmap(
            raw_tiff_path, 
            output_dir, 
            location_name,
            elevation_range={
                'min': worldmap_info['min_elev'],
                'max': worldmap_info['max_elev']
            }
        )
        
        if heightmap_info:
            print(f"\nHeightmap processing complete!")
            print(f"Both heightmaps are normalized to the full area's elevation range:")
            print(f"- Elevation range: {heightmap_info['min_elev']:.2f}m to {heightmap_info['max_elev']:.2f}m")
            print(f"- Total relief: {heightmap_info['range']:.2f}m")
            print(f"\nFiles created:")
            print(f"- Base Heightmap PNG: {heightmap_info['png']}")
            print(f"- Base Heightmap TIFF: {heightmap_info['tiff']}")
            
            # Add worldmap info to heightmap_info
            heightmap_info['worldmap'] = {
                'png': worldmap_info['png'],
                'tiff': worldmap_info['tiff']
            }
            print(f"- Worldmap PNG: {heightmap_info['worldmap']['png']}")
            print(f"- Worldmap TIFF: {heightmap_info['worldmap']['tiff']}")
            
            # Visualize if requested
            if visualize:
                visualize_heightmaps(heightmap_info, lat, lon, output_dir)
            
            print("\nHeightmaps are ready for use in Cities: Skylines II")
            print(f"Files are located in: {os.path.abspath(output_dir)}")
        else:
            print("Heightmap processing failed.")
            
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate heightmaps for Cities Skylines 2 (Continental US Only)')
    parser.add_argument('input', type=str, help='Continental US place name or "latitude,longitude" coordinates')
    parser.add_argument('--output', type=str, default='elevation_output', help='Output directory')
    parser.add_argument('--no-visualization', action='store_true', help='Skip visualization')
    return parser.parse_args()

if __name__ == "__main__":
    # Check if arguments were provided
    if len(sys.argv) < 2:
        print("Usage: python main.py <place_name_or_coordinates> [--output directory] [--no-visualization]")
        print("\nNOTE: This tool only works for locations within the continental United States.")
        print("\nExamples:")
        print("  python main.py \"Grand Canyon\"")
        print("  python main.py \"40.7128,-74.0060\"")
        sys.exit(1)
    
    args = parse_arguments()
    main(args.input, args.output, not args.no_visualization)
