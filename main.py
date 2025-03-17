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
    """Main function to process either place name or coordinates and generate heightmap."""
    try:
        print("\n=== Cities Skylines II Heightmap Generator ===")
        
        os.makedirs(output_dir, exist_ok=True)
        coordinates = parse_coordinates(input_data)
        
        if coordinates:
            lat, lon = coordinates
            if not is_in_continental_us(lat, lon):
                print(f"Error: Coordinates ({lat}, {lon}) are outside the continental United States.")
                return
            location_name = None
        else:
            print(f"Looking up '{input_data}'...")
            coordinates = geocode_place_name(input_data)
            if not coordinates:
                print(f"Could not find coordinates for '{input_data}'")
                return
            lat, lon = coordinates
            if not is_in_continental_us(lat, lon):
                print(f"Error: '{input_data}' is outside the continental United States.")
                return
            location_name = input_data
        
        print(f"\nGenerating heightmap for: {lat}, {lon}")
        
        # Fetch and process worldmap first
        print("\n1. Downloading data for extended area (57.344 x 57.344 km)...")
        worldmap_elevation_data = fetch_elevation_data(f"{lat},{lon}", for_worldmap=True)
        
        if not worldmap_elevation_data or 'data_path' not in worldmap_elevation_data:
            print("Failed to fetch worldmap elevation data.")
            return
        
        worldmap_info = process_heightmap(
            worldmap_elevation_data['data_path'], 
            output_dir, 
            f"{location_name}_worldmap" if location_name else "worldmap"
        )
        
        if not worldmap_info:
            print("Failed to process worldmap.")
            return
        
        # Now fetch and process core heightmap
        print("\n2. Downloading high-resolution data for playable area (14.336 x 14.336 km)...")
        core_elevation_data = fetch_elevation_data(f"{lat},{lon}", for_worldmap=False)
        
        if not core_elevation_data or 'data_path' not in core_elevation_data:
            print("Failed to fetch elevation data for core area")
            return
        
        heightmap_info = process_heightmap(
            core_elevation_data['data_path'], 
            output_dir, 
            location_name,
            elevation_range={
                'min': worldmap_info['min_elev'],
                'max': worldmap_info['max_elev']
            }
        )
        
        if heightmap_info:
            print("\n=== Processing Complete ===")
            print(f"Elevation range: {heightmap_info['min_elev']:.2f}m to {heightmap_info['max_elev']:.2f}m")
            print(f"Total relief: {heightmap_info['range']:.2f}m")
            
            # Add worldmap info
            heightmap_info['worldmap'] = {
                'png': worldmap_info['png'],
                'tiff': worldmap_info['tiff']
            }
            
            if visualize:
                visualize_heightmaps(heightmap_info, lat, lon, output_dir)
            
            print(f"\nFiles are in: {os.path.abspath(output_dir)}")
        else:
            print("Heightmap processing failed.")
            
    except Exception as e:
        print(f"Error: {str(e)}")

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
