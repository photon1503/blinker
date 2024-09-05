import os
import cv2
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import concurrent.futures
from collections import OrderedDict
from LRUcache import *
import sys


reference_dec = None
reference_ra = None

def stf_sigma_stretch(image_data, sigma=2.8, clip=True):
    """
    Apply a sigma-based stretch to the image data.
    
    :param image_data: The input image data from the FITS file.
    :param sigma: The number of standard deviations to include in the stretch.
    :param clip: Whether to clip the output to the range [0, 1].
    :return: The stretched image data normalized to [0, 255].
    """
    # Calculate mean and standard deviation
    mean = np.mean(image_data)
    std = np.std(image_data)
    
    # Define the lower and upper bounds for the stretch
    lower_bound = mean - 0.8 * std
    upper_bound = mean + sigma * std
    
    # Apply the stretch
    stretched_data = (image_data - lower_bound) / (upper_bound - lower_bound)
    
    if clip:
        stretched_data = np.clip(stretched_data, 0, 1)
    
    # Normalize to the 0-255 range
    stretched_data = stretched_data * 255
    
    return stretched_data.astype(np.uint8)

def shift_image_based_on_dec_ra(image_data, header):
    global reference_dec, reference_ra
    
    # Extract DEC and RA from header
    current_dec = header['DEC']
    current_ra = header['RA']
    
    # Extract pixel size from header (micrometers)
    pixel_size_x = header.get('XPIXSZ', 1)  # Pixel size in micrometers
    pixel_size_y = header.get('YPIXSZ', 1)  # Pixel size in micrometers
    
    # Extract focal length from header (millimeters)
    focal_length = header.get('FOCALLEN', 1)  # Focal length in millimeters
    
    # Convert pixel size to arcseconds per pixel
    pixel_scale_x = (pixel_size_x / focal_length) * 206.265   # Arcseconds per pixel
    pixel_scale_y = (pixel_size_y / focal_length) * 206.265   # Arcseconds per pixel
    
    print(f"Pixel Scale X: {pixel_scale_x} arcsec/pixel")
    print(f"Pixel Scale Y: {pixel_scale_y} arcsec/pixel")
    
    if reference_dec is None and reference_ra is None:
        # Store the DEC and RA of the first image as reference
        reference_dec = current_dec
        reference_ra = current_ra
    
    # Calculate the shift needed in arcseconds
    shift_dec_arcsec = (current_dec - reference_dec) * 3600  # Convert degrees to arcseconds
    shift_ra_arcsec = (current_ra - reference_ra) * 3600  # Convert degrees to arcseconds
    
    print(f"Shift DEC (arcsec): {shift_dec_arcsec}")
    print(f"Shift RA (arcsec): {shift_ra_arcsec}")
    
    # Convert the shift from arcseconds to pixels
    shift_y = int(shift_dec_arcsec / pixel_scale_y)
    shift_x = int(shift_ra_arcsec / pixel_scale_x)
    
    print(f"Shift Y (pixels): {shift_y}")
    print(f"Shift X (pixels): {shift_x}")
    
    # Apply the shift to the image data
    shifted_data = np.roll(image_data, shift_y, axis=0)
    shifted_data = np.roll(shifted_data, shift_x, axis=1)
    
    return shifted_data

def read_and_stretch_fits(file_path, sigma=2.0):
    # Read FITS file
    with fits.open(file_path) as hdul:
        image_data = hdul[0].data
        header = hdul[0].header  # Get the FITS header
    
    # resize to HD
    image_data = cv2.resize(image_data, (1920, 1080))

    # Stretch the image using STF sigma scaling
    stretched_data = stf_sigma_stretch(image_data, sigma=sigma)
    
     # Fix orientation based on header
    oriented_data = fix_image_orientation_with_header(stretched_data, header)
    
    #shifted_data = shift_image_based_on_dec_ra(oriented_data, header)
    
    return oriented_data


def fix_image_orientation_with_header(image_data, header):
    """
    Rotate the image based on the orientation information in the FITS header.
    
    :param image_data: The input image data.
    :param header: The FITS header.
    :return: The correctly oriented image.
    """

    # Determine the rotation needed (round to nearest 90 degrees)
    if header['PIERSIDE'] != "East":
      
            rotated_data = cv2.rotate(image_data, cv2.ROTATE_180)
    else:
        rotated_data = image_data

    
    


    
    
    return rotated_data

def display_fits_as_film(folder_path, delay=100):
    # Get a list of FITS files in the folder
    fits_files = [f for f in os.listdir(folder_path) if f.endswith('.fits')]
    
    if not fits_files:
        print("No FITS files found in the folder.")
        return
    
    # Sort files for a proper sequence
    fits_files.sort()

    # Set up control variables
    paused = False
    index = 0
    num_files = len(fits_files)
    
        # Set up the LRU cache with a 16 GB limit
    max_cache_size = 16 * 1024 * 1024 * 1024  # 16 GB
    cache = LRUCache(max_cache_size)
    
        
  # Use a ThreadPoolExecutor for read-ahead
        
    while True:
        current_file = os.path.join(folder_path, fits_files[index])

        # Check if the current image is in cache
        image_data = cache.get(current_file)
        if image_data is None:
            # Load and process the current image if not in cache
            image_data = read_and_stretch_fits(current_file)
            # Store the processed image in cache
            cache.put(current_file, image_data)
        
        # Overlay the filename on the image
        filename = fits_files[index]
        image_with_overlay = overlay_filename(image_data, filename)


        
        # Display the image with the overlay
        cv2.imshow('FITS Film', image_with_overlay)
        
        # Handle key events
        key = cv2.waitKey(delay if not paused else 0) & 0xFF
        
        if key == ord(' '):  # Spacebar to pause/play
            paused = not paused
        elif key == 27:  # ESC to exit
            break
        elif key == 82 or key == ord('r'):  # 'R' key to reset to first frame
            index = 0
        elif key == ord('p'):
            # move the file to a different folder
            # create path if it does not exist
            if not os.path.exists(os.path.join(folder_path, "BAD")):
                os.makedirs(os.path.join(folder_path, "BAD"))
            os.rename(current_file, os.path.join(folder_path, "BAD",  fits_files[index]))
            del fits_files[index]
            if current_file in cache:
                del cache.cache[current_file]  # Remove from cache as well
            num_files -= 1
            if index >= num_files:
                index = num_files - 1
            if num_files == 0:
                break
        elif key == ord('d'):  # Right arrow to go forward
            index = (index + 1) % num_files
        elif key == ord('a'):  # Left arrow to go back
            index = (index - 1) % num_files
        

        
        if not paused and num_files > 1:
            # Pre-fetch the next image in the background, but check the cache first
            index=(index + 1) % num_files

        
    # Clean up windows
    cv2.destroyAllWindows()

def overlay_filename(image_data, filename):

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (255, 255, 255)  # White text
    thickness = 2
    position = (10, 30)  # Top-left corner

    cv2.putText(image_data, filename, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image_data


if __name__ == "__main__":
    # get folder from argument
    folder_path = sys.argv[1]
    display_fits_as_film(folder_path, delay=1)  # Adjust delay as needed

