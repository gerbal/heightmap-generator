import numpy as np
from PIL import Image

def generate_world_map(elevation_data):
    # Ensure the elevation data is in the correct format
    elevation_array = np.array(elevation_data, dtype=np.uint16)
    
    # Resize the elevation data to cover 57344 meters
    world_map_size = (57344, 57344)
    world_map = np.resize(elevation_array, world_map_size)
    
    # Create an image from the world map data
    world_map_image = Image.fromarray(world_map, mode='I;16')
    
    # Convert the image to grayscale
    world_map_image = world_map_image.convert('L')
    
    # Save the world map image as a .png file
    world_map_image.save('world_map.png')

    return world_map_image
