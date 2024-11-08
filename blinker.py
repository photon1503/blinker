import os
import cv2
import numpy as np
from astropy.io import fits
from LRUcache import *
import sys
from concurrent.futures import ThreadPoolExecutor
import time
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import tkinter as tk
from tkinter import filedialog





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
    lower_bound = mean - 0.3 * std
    upper_bound = mean + sigma * std
    
    # Apply the stretch
    stretched_data = (image_data - lower_bound) / (upper_bound - lower_bound)
    
    if clip:
        stretched_data = np.clip(stretched_data, 0, 1)
    
    # Normalize to the 0-255 range
    stretched_data = stretched_data * 255
    
    return stretched_data.astype(np.uint8)

def shift_image(image_data, header):
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

def read_and_stretch(file_path, sigma=2.0):
    # Read FITS file
    with fits.open(file_path) as hdul:
        image_data = hdul[0].data
        header = hdul[0].header  # Get the FITS header
    
    # remove hot pixels
    image_data = cv2.medianBlur(image_data, 3)

    # resize to HD
    image_data = cv2.resize(image_data, (1920, 1080))

    # Stretch the image using STF sigma scaling
    stretched_data = stf_sigma_stretch(image_data, sigma=sigma)
    
     # Fix orientation based on header
    oriented_data = rotate(stretched_data, header)
    
    #shifted_data = shift_image_based_on_dec_ra(oriented_data, header)
    
    return oriented_data


def rotate(image_data, header):
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

def overlay_progress_bar(image_data, current_index, total_images):
    overlay = image_data.copy()
    bar_width = int((current_index / total_images) * overlay.shape[1])
    cv2.rectangle(overlay, (0, overlay.shape[0] - 20), (bar_width, overlay.shape[0]), (255, 255, 255), -1)
    cv2.putText(overlay, f'{current_index + 1}/{total_images}', (10, overlay.shape[0] - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return overlay

def preload_images(fits_files, folder_path, cache):
    for fits_file in fits_files:
        current_file = os.path.join(folder_path, fits_file)
        if cache.get(current_file) is None:
            image_data = read_and_stretch(current_file)
            image_with_overlay = overlay_filename(image_data, fits_file)
            cache.put(current_file, image_with_overlay)

def display_images(folder_path, delay=100):
    # Get a list of FITS files in the folder

    #get all fits or fts or fit
    fits_files = [f for f in os.listdir(folder_path) if f.endswith('.fits') or f.endswith('.fts') or f.endswith('.fit')]
    
    if not fits_files:
        print("No FITS files found in the folder.")
        return
    
    # Sort files for a proper sequence
    fits_files.sort()

    # Set up control variables

    total_images = len(fits_files)
    
        # Set up the LRU cache with a 16 GB limit
    max_cache_size = 16 * 1024 * 1024 * 1024  # 16 GB
    cache = LRUCache(max_cache_size)
    
        
  # Use a ThreadPoolExecutor for read-ahead
    with ThreadPoolExecutor() as executor:    
        paused = False
        index = 0
        zoom = False
        zoom_factor = 3
        zoom_size = 100

        executor.submit(preload_images, fits_files, folder_path, cache)

                # Initialize pygame
        pygame.init()
 # Get the dimensions of the first image to set the window size
        first_image_path = os.path.join(folder_path, fits_files[0])
        first_image_data = read_and_stretch(first_image_path)
        height, width = first_image_data.shape[:2]
        
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('FITS Blinker')
        
        clock = pygame.time.Clock()

        while True:
            start_time = time.time()

            current_file = os.path.join(folder_path, fits_files[index])
            image_data = cache.get(current_file)

            if image_data is None:
                image_data = read_and_stretch(current_file)
                image_with_overlay = overlay_filename(image_data, fits_files[index])
                cache.put(current_file, image_with_overlay)
                image_data = image_with_overlay
            
            image_with_progress = overlay_progress_bar(image_data, index, total_images)
              # Ensure the image data has three dimensions
            if len(image_with_progress.shape) == 2:
                image_with_progress = np.expand_dims(image_with_progress, axis=2)
                image_with_progress = np.repeat(image_with_progress, 3, axis=2)
            
            # Transpose the image to correct the rotation
            image_with_progress = np.transpose(image_with_progress, (1, 0, 2))

            # Convert image to pygame surface
            image_surface = pygame.surfarray.make_surface(cv2.cvtColor(image_with_progress, cv2.COLOR_BGR2RGB))
            screen.blit(image_surface, (0, 0))

            # Handle zoom
            if zoom:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                zoom_rect = pygame.Rect(mouse_x - zoom_size // 2, mouse_y - zoom_size // 2, zoom_size, zoom_size)
                zoom_surface = pygame.Surface((zoom_size, zoom_size), pygame.SRCALPHA)
                zoom_surface.blit(image_surface, (0, 0), zoom_rect)
                zoom_surface = pygame.transform.scale(zoom_surface, (zoom_size * zoom_factor, zoom_size * zoom_factor))
                
                # Create a circular mask
                mask = pygame.Surface((zoom_size * zoom_factor, zoom_size * zoom_factor), pygame.SRCALPHA)
                pygame.draw.circle(mask, (255, 255, 255, 255), (zoom_size * zoom_factor // 2, zoom_size * zoom_factor // 2), zoom_size * zoom_factor // 2)
                
                # Apply the mask to the zoom surface
                zoom_surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
                
                # Draw the zoomed circular area on the screen
                screen.blit(zoom_surface, (mouse_x - zoom_size * zoom_factor // 2, mouse_y - zoom_size * zoom_factor // 2))
                
                # Draw the boundary
                pygame.draw.circle(screen, (255, 0, 0), (mouse_x, mouse_y), zoom_size * zoom_factor // 2, 2)  # Red boundary with thickness 2


            pygame.display.flip()
            
            # Handle key events
            #key = cv2.waitKey(5) & 0xFF
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:  # Spacebar to pause/play
                        paused = not paused
                    elif event.key == pygame.K_ESCAPE:  # ESC to exit
                        pygame.quit()
                        return
                    elif event.key == pygame.K_r:  # 'R' key to reset to first frame
                        index = 0
                    elif event.key == pygame.K_p:
                        # move the file to a different folder
                        # create path if it does not exist
                        if not os.path.exists(os.path.join(folder_path, "BAD")):
                            os.makedirs(os.path.join(folder_path, "BAD"))
                        os.rename(current_file, os.path.join(folder_path, "BAD",  fits_files[index]))
                        del fits_files[index]
                        if current_file in cache:
                            del cache.cache[current_file]  # Remove from cache as well
                        total_images -= 1
                        if index >= total_images:
                            index = total_images - 1
                        if total_images == 0:
                            pygame.quit()
                            return
                    elif event.key == pygame.K_d:  # Right arrow to go forward
                        index = (index + 1) % total_images
                    elif event.key == pygame.K_a:  # Left arrow to go back
                        index = (index - 1) % total_images
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 3:  # Right mouse button
                        zoom = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 3:  # Right mouse button
                        zoom = False
            

            
            if not paused and total_images > 1:
                index=(index + 1) % total_images

            elapsed_time = time.time() - start_time
            # Adjust the delay to maintain a consistent frame rate
            time_to_wait = max(1, int(delay - elapsed_time * 1000))
            clock.tick(1000 // time_to_wait)


        
    # Clean up windows
    pygame.quit()

def overlay_filename(image_data, filename):

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (255, 255, 255)  # White text
    thickness = 2
    position = (10, 30)  # Top-left corner

    cv2.putText(image_data, filename, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image_data


if __name__ == "__main__":
    folderpath=None
    if len(sys.argv) > 1:
        folderpath = sys.argv[1]
    else:
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        folderpath = filedialog.askdirectory(title="Select Folder Containing FITS Files")
        
    if folderpath:
        display_images(folderpath, delay=10)  # Adjust delay as needed
    else:
        print("No folder selected.")

