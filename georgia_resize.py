#!/usr/bin/env python3

import os
import cv2
import numpy as np

def resize_and_pad(image, output_size=(384, 286)):
    original_size = (image.shape[1], image.shape[0])
    scaling_factor = min(output_size[0] / original_size[0], output_size[1] / original_size[1])
    new_size = (int(original_size[0] * scaling_factor), int(original_size[1] * scaling_factor))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    new_image = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    start_x = (output_size[0] - new_size[0]) // 2
    start_y = (output_size[1] - new_size[1]) // 2
    new_image[start_y:start_y+new_size[1], start_x:start_x+new_size[0]] = resized_image
    return new_image

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]
    for subfolder in subfolders:
        image_files = [f for f in os.listdir(subfolder) if f.endswith('.jpg')]
        for image_file in image_files:
            image_path = os.path.join(subfolder, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                resized_padded_image = resize_and_pad(image)
                output_path = os.path.join(output_folder, os.path.basename(subfolder) + '_' + image_file)
                cv2.imwrite(output_path, resized_padded_image)
                print(f'Processed and saved: {output_path}')
            else:
                print(f'Failed to read image: {image_path}')

# Path configuration
input_folder = 'gt_db'
output_folder = 'resized_gt_db'

# Process all images
process_images(input_folder, output_folder)
