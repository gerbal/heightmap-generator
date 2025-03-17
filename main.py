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
import numpy as np  # Adding numpy import which was missing
from heightmap_generator import process_heightmap, generate_worldmap, visualize_heightmaps, calculate_bounding_box
from fetch_bathymetry import fetch_bathymetry_data, apply_bathymetry_to_heightmap, visualize_bathymetry

def main(input_data, output_dir="elevation_output", visualize=True, with_bathymetry=False, bathymetry_source=None):
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
        
        if not heightmap_info:
            print("Heightmap processing failed.")
            return
        
        # Add worldmap info
        heightmap_info['worldmap'] = {
            'png': worldmap_info['png'],
            'tiff': worldmap_info['tiff']
        }
        
        # Process with bathymetry if requested
        if with_bathymetry:
            print("\n3. Fetching and applying bathymetry data...")
            
            # For the core heightmap (playable area)
            core_bbox = calculate_bounding_box(lat, lon, width_km=14.336, height_km=14.336)
            
            # Custom source list if specified, otherwise use default
            sources = [bathymetry_source] if bathymetry_source else None
            
            # Fetch bathymetry data
            bathymetry_data, bathy_meta = fetch_bathymetry_data(
                core_bbox, 
                output_dir=output_dir,
                sources=sources,
                resolution=4096  # Match heightmap resolution
            )
            
            if bathymetry_data is not None:
                source_name = bathy_meta.get('source_name', 'unknown')
                print(f"Successfully fetched bathymetry data from {source_name}")
                
                # Create a visualization of the bathymetry data
                viz_path = visualize_bathymetry(bathymetry_data, bathy_meta, output_dir)
                heightmap_info['bathymetry_viz'] = viz_path
                
                # Apply bathymetry to heightmap
                # First, we need to identify water areas in the heightmap
                with_topo_suffix = f"_topo_{bathy_meta['source']}"
                
                # Get parsed elevation data
                from fetch_elevation_data import parse_elevation_data
                heightmap_parsed_data = parse_elevation_data(core_elevation_data['data_path'])
                
                # Create a water mask where elevation is close to or below sea level
                water_mask = heightmap_parsed_data <= 0
                
                if water_mask.any():
                    print(f"Applying bathymetry to heightmap ({np.sum(water_mask)} water pixels)")
                    
                    # Apply bathymetry to core heightmap
                    heightmap_with_bathy = apply_bathymetry_to_heightmap(
                        heightmap_parsed_data.copy(),
                        water_mask,
                        bathymetry_data,
                        bathy_meta,
                        heightmap_info['min_elev'],
                        max_depth=-30,  # Maximum water depth in meters
                        coastal_depth=-1  # Depth at coastlines in meters
                    )
                    
                    # Process the heightmap with bathymetry
                    bathymetry_heightmap_info = process_heightmap(
                        None,  # No file path since we're passing data directly
                        output_dir,
                        f"{location_name}{with_topo_suffix}" if location_name else f"heightmap{with_topo_suffix}",
                        elevation_range={
                            'min': min(heightmap_info['min_elev'], np.min(heightmap_with_bathy)),
                            'max': heightmap_info['max_elev']
                        },
                        provided_data=heightmap_with_bathy
                    )
                    
                    if bathymetry_heightmap_info:
                        print(f"Successfully created heightmap with bathymetry data")
                        heightmap_info['with_bathymetry'] = bathymetry_heightmap_info
                    
                    # Also apply to worldmap if available
                    if heightmap_info['worldmap']:
                        # Process worldmap with bathymetry data
                        worldmap_parsed_data = parse_elevation_data(worldmap_elevation_data['data_path'])
                        worldmap_water_mask = worldmap_parsed_data <= 0
                        
                        # For the worldmap, we might need to get broader bathymetry data
                        worldmap_bbox = calculate_bounding_box(lat, lon, width_km=57.344, height_km=57.344)
                        
                        # If bounding box is significantly different, fetch new bathymetry data for it
                        worldmap_bathy_data = bathymetry_data
                        worldmap_bathy_meta = bathy_meta
                        
                        worldmap_with_bathy = apply_bathymetry_to_heightmap(
                            worldmap_parsed_data.copy(),
                            worldmap_water_mask,
                            worldmap_bathy_data,
                            worldmap_bathy_meta,
                            worldmap_info['min_elev'],
                            max_depth=-30,
                            coastal_depth=-1
                        )
                        
                        # Process the worldmap with bathymetry
                        worldmap_base_name = f"{location_name}_worldmap" if location_name else "worldmap"
                        worldmap_bathy_name = f"{worldmap_base_name}{with_topo_suffix}"
                        
                        worldmap_bathy_info = process_heightmap(
                            None,
                            output_dir,
                            worldmap_bathy_name,
                            elevation_range={
                                'min': min(worldmap_info['min_elev'], np.min(worldmap_with_bathy)),
                                'max': worldmap_info['max_elev']
                            },
                            provided_data=worldmap_with_bathy
                        )
                        
                        if worldmap_bathy_info:
                            print(f"Successfully created worldmap with bathymetry data")
                            heightmap_info['worldmap_with_bathymetry'] = worldmap_bathy_info
                else:
                    print("No water areas found in heightmap, bathymetry will not be applied")
            else:
                print("Failed to fetch bathymetry data, continuing without it")
        
        print("\n=== Processing Complete ===")
        print(f"Elevation range: {heightmap_info['min_elev']:.2f}m to {heightmap_info['max_elev']:.2f}m")
        print(f"Total relief: {heightmap_info['range']:.2f}m")
        
        if visualize:
            visualize_heightmaps(heightmap_info, lat, lon, output_dir)
        
        print(f"\nFiles are in: {os.path.abspath(output_dir)}")
        return heightmap_info
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate heightmaps for Cities Skylines 2 (Continental US Only)')
    parser.add_argument('input', type=str, help='Continental US place name or "latitude,longitude" coordinates')
    parser.add_argument('--output', type=str, default='elevation_output', help='Output directory')
    parser.add_argument('--no-visualization', action='store_true', help='Skip visualization')
    parser.add_argument('--with-bathymetry', action='store_true', help='Apply bathymetry data to water areas')
    parser.add_argument('--bathymetry-source', type=str, choices=['noaa_dem', 'noaa_coastal', 'noaa_ninth_arc', 'gebco', 'noaa_crm'],
                       help='Specify bathymetry data source (default: auto-select best available)')
    return parser.parse_args()

if __name__ == "__main__":
    # Check if arguments were provided
    if len(sys.argv) < 2:
        print("Usage: python main.py <place_name_or_coordinates> [--output directory] [--no-visualization] [--with-bathymetry] [--bathymetry-source SOURCE]")
        print("\nNOTE: This tool only works for locations within the continental United States.")
        print("\nExamples:")
        print("  python main.py \"Grand Canyon\"")
        print("  python main.py \"40.7128,-74.0060\"")
        print("  python main.py \"New York\" --with-bathymetry")
        print("  python main.py \"San Francisco\" --with-bathymetry --bathymetry-source gebco")
        sys.exit(1)
    
    args = parse_arguments()
    main(args.input, args.output, not args.no_visualization, args.with_bathymetry, args.bathymetry_source)
