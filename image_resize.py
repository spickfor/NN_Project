#!/usr/bin/env python3

import cv2
import numpy as np

def resize_and_pad(image_path, output_size=(384, 286)):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Get the original size of the image
    original_size = (image.shape[1], image.shape[0])

    # Calculate the scaling factor
    scaling_factor = min(output_size[0] / original_size[0], output_size[1] / original_size[1])

    # Calculate the new size and resize the image
    new_size = (int(original_size[0] * scaling_factor), int(original_size[1] * scaling_factor))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    # Create a new image with the desired output size and paste the resized image into it
    new_image = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    start_x = (output_size[0] - new_size[0]) // 2
    start_y = (output_size[1] - new_size[1]) // 2
    new_image[start_y:start_y+new_size[1], start_x:start_x+new_size[0]] = resized_image

    return new_image

# Use the function on the provided image
resized_padded_image = resize_and_pad('test.jpeg')
cv2.imshow('Resized and Padded Image', resized_padded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally save the processed image
cv2.imwrite('resized_padded_test.jpeg', resized_padded_image)
