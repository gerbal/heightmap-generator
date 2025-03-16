import numpy as np
from PIL import Image

def convert_to_heightmap(elevation_data):
    # Ensure the elevation data is in the correct format
    elevation_array = np.array(elevation_data, dtype=np.uint16)
    
    # Resize the elevation data to 4096x4096 pixels
    heightmap = np.resize(elevation_array, (4096, 4096))
    
    # Create an image from the heightmap data
    heightmap_image = Image.fromarray(heightmap, mode='I;16')
    
    # Save the heightmap image as a .png file
    heightmap_image.save('heightmap.png')

    return heightmap_image
