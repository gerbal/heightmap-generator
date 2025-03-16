#!/usr/bin/env python3
"""
Test script to fetch and process bathymetry data for Manhattan.

This script focuses specifically on testing the bathymetry fetching functionality
for the Manhattan area to ensure proper water depth representation.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import hashlib
import argparse
from fetch_bathymetry import (
    fetch_bathymetry_data,
    visualize_bathymetry,
    apply_bathymetry_to_heightmap,
    BATHYMETRY_SOURCES
)
from heightmap_generator import calculate_bounding_box

def test_bathymetry_source(source_name, bbox, resolution=4096, output_dir="elevation_output", force_refresh=False):
    """Test a specific bathymetry data source.
    
    Args:
        source_name (str): Name of the bathymetry source to test
        bbox (tuple): Bounding box as (min_lon, min_lat, max_lon, max_lat)
        resolution (int): Resolution of the requested bathymetry data
        output_dir (str): Directory to save output files
        force_refresh (bool): If True, ignore cached data and fetch fresh data
        
    Returns:
        tuple: (numpy.ndarray, dict) - Bathymetry data and metadata if successful, or (None, None)
    """
    print(f"\n=== Testing bathymetry source: {source_name} ===")
    print(f"Source details: {BATHYMETRY_SOURCES.get(source_name, {}).get('description', 'Unknown source')}")
    print(f"Resolution: {resolution}x{resolution}")
    
    # Generate cache filename based on source, bbox and resolution
    bbox_str = f"{source_name}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_{resolution}_bathy"
    bbox_hash = hashlib.md5(bbox_str.encode()).hexdigest()
    cache_file_path = os.path.join(output_dir, f"bathymetry_{bbox_hash}.tiff")
    metadata_file_path = os.path.join(output_dir, f"bathymetry_{bbox_hash}_meta.json")
    
    # Remove existing cache if requested
    if force_refresh:
        if os.path.exists(cache_file_path):
            os.remove(cache_file_path)
            print(f"Removed cached file: {cache_file_path}")
        if os.path.exists(metadata_file_path):
            os.remove(metadata_file_path)
            print(f"Removed cached metadata: {metadata_file_path}")
    
    # Time the fetch operation
    start_time = time.time()
    
    # Fetch bathymetry data from the specified source
    bathymetry_data, bathy_meta = fetch_bathymetry_data(
        bbox, 
        output_dir=output_dir,
        sources=[source_name],
        resolution=resolution,
        cache=not force_refresh
    )
    
    elapsed_time = time.time() - start_time
    
    # Analyze and visualize the results
    if bathymetry_data is not None:
        print(f"Successfully fetched bathymetry data in {elapsed_time:.2f} seconds:")
        print(f"- Shape: {bathymetry_data.shape}")
        print(f"- Min depth: {np.nanmin(bathymetry_data):.2f}m")
        print(f"- Max depth: {np.nanmax(bathymetry_data):.2f}m")
        print(f"- Mean depth: {np.nanmean(bathymetry_data):.2f}m")
        print(f"- Source: {bathy_meta.get('source_name', 'Unknown')}")
        
        # Count non-NaN values as a measure of data completeness
        non_nan_count = np.count_nonzero(~np.isnan(bathymetry_data))
        data_completeness = non_nan_count / bathymetry_data.size * 100
        print(f"- Data completeness: {data_completeness:.1f}% ({non_nan_count} of {bathymetry_data.size} pixels)")
        
        # Create a filename with source and resolution for easy comparison
        viz_filename = f"manhattan_{source_name}_{resolution}_bathymetry.png"
        viz_path = os.path.join(output_dir, viz_filename)
        
        # Create a visualization
        plt.figure(figsize=(15, 12))
        
        # Create a custom colormap optimized for bathymetry
        cmap = plt.cm.viridis.copy()
        
        # Create a masked array to handle NaN values
        masked_data = np.ma.masked_invalid(bathymetry_data)
        
        # Plot the bathymetry data with custom colormap
        img = plt.imshow(masked_data, cmap=cmap)
        plt.colorbar(img, label='Depth (m)')
        
        # Add contour lines at specific depths
        contour_levels = [-100, -50, -30, -20, -15, -10, -7, -5, -3, -2, -1]
        
        # Filter levels to only include those in our data range
        min_depth = np.nanmin(bathymetry_data)
        max_depth = np.nanmax(bathymetry_data)
        valid_levels = [level for level in contour_levels if min_depth <= level <= max_depth]
        
        if valid_levels:
            contours = plt.contour(masked_data, levels=valid_levels, colors='white', alpha=0.6)
            plt.clabel(contours, inline=True, fontsize=9, fmt='%d m')
        
        # Add a title with source and resolution information
        title = f"Manhattan Bathymetry: {BATHYMETRY_SOURCES.get(source_name, {}).get('name', source_name)}\n"
        title += f"Resolution: {bathymetry_data.shape[0]}x{bathymetry_data.shape[1]} pixels\n"
        title += f"Depth Range: {min_depth:.1f}m to {max_depth:.1f}m (avg: {np.nanmean(bathymetry_data):.1f}m)"
        plt.title(title)
        
        plt.tight_layout()
        plt.savefig(viz_path, dpi=300)
        plt.close()
        
        print(f"Saved bathymetry visualization to {viz_path}")
        
        return bathymetry_data, bathy_meta
    else:
        print(f"Failed to fetch bathymetry data from {source_name}.")
        return None, None

def test_manhattan_bathymetry():
    """Test fetching and visualizing bathymetry data for Manhattan."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test bathymetry sources for Manhattan")
    parser.add_argument("--source", help="Specify a single bathymetry source to test", default=None)
    parser.add_argument("--resolution", type=int, help="Resolution in pixels", default=4096)
    parser.add_argument("--refresh", action="store_true", help="Force refresh cached data")
    parser.add_argument("--width-km", type=float, help="Width of bounding box in km", default=14.336)
    parser.add_argument("--height-km", type=float, help="Height of bounding box in km", default=14.336)
    args = parser.parse_args()
    
    # Manhattan coordinates (approximately centered on Lower Manhattan / NY Harbor)
    manhattan_lat = 40.7128
    manhattan_lon = -74.0060
    output_dir = "elevation_output"
    
    print(f"Testing bathymetry data for Manhattan at {manhattan_lat}, {manhattan_lon}")
    
    # Calculate a bounding box for Manhattan and surrounding waters
    bbox = calculate_bounding_box(manhattan_lat, manhattan_lon, 
                                  width_km=args.width_km, 
                                  height_km=args.height_km)
    print(f"Bounding box: {bbox}")
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which sources to test
    if args.source:
        if args.source in BATHYMETRY_SOURCES:
            sources_to_test = [args.source]
        else:
            print(f"Unknown source: {args.source}")
            print(f"Available sources: {', '.join(BATHYMETRY_SOURCES.keys())}")
            return
    else:
        # Test all sources
        sources_to_test = list(BATHYMETRY_SOURCES.keys())
    
    # Dictionary to store results for comparison
    results = {}
    
    # Test each source
    for source in sources_to_test:
        data, meta = test_bathymetry_source(
            source, 
            bbox, 
            resolution=args.resolution, 
            output_dir=output_dir,
            force_refresh=args.refresh
        )
        
        if data is not None:
            # Store basic metrics for comparison
            results[source] = {
                "min_depth": float(np.nanmin(data)),
                "max_depth": float(np.nanmax(data)),
                "mean_depth": float(np.nanmean(data)),
                "non_nan_percentage": float(np.count_nonzero(~np.isnan(data)) / data.size * 100),
                "shape": list(data.shape)
            }
    
    # Save comparison results
    if results:
        comparison_path = os.path.join(output_dir, f"manhattan_bathymetry_comparison_{args.resolution}.json")
        with open(comparison_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n=== Bathymetry Source Comparison ===")
        print(f"Results saved to: {comparison_path}")
        
        # Print a simple comparison table
        print("\nSource               | Resolution | Min Depth | Max Depth | Mean Depth | Coverage %")
        print("-" * 85)
        for source, metrics in results.items():
            source_name = BATHYMETRY_SOURCES.get(source, {}).get('name', source)
            print(f"{source_name[:20]:<20} | {metrics['shape'][0]}x{metrics['shape'][1]:<8} | "
                  f"{metrics['min_depth']:9.2f} | {metrics['max_depth']:9.2f} | "
                  f"{metrics['mean_depth']:10.2f} | {metrics['non_nan_percentage']:9.1f}%")
    else:
        print("No successful bathymetry data fetches.")

if __name__ == "__main__":
    test_manhattan_bathymetry()