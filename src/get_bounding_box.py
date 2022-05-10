# Built in libraries
import os
import sys
import time
import argparse
from pathlib import Path
import concurrent.futures
from os.path import abspath, join

# External libraries
import numpy as np
import pandas as pd
from skimage import io
import cv2

def get_options():
    # Script description
    description="""Converts a nuclear into images for each single nuclei"""

    # Add parser
    parser = argparse.ArgumentParser(description)

    # Tool Input
    input = parser.add_argument_group(title="Input")
    input.add_argument("-i", "--image", dest="image", action="store", required=True, help="Pathway to input image.")
    input.add_argument("-m", "--mask", dest="mask", action="store", required=True, help="Pathway to mask segmentation.")
    input.add_argument("-c", "--channel", dest="channel", action="store", required=True, help="DAPI channel.")

    # Tool output
    output = parser.add_argument_group(title="Output")
    output.add_argument("-o", "--out", dest="out", action="store", required=True, help="Path to the output data folder")

    # Parse arguments
    args = parser.parse_args()

    # Standardize paths
    args.data = abspath(args.data)
    args.out = abspath(args.out)

    return args


def im_min_max_scaler(im, minmaxrange=(0, 1)):
    # Unpack value range
    min, max = minmaxrange

    # Normal scaling (0 - 1)
    im = (im - im.min()) / (im.max() - im.min())

    # In case of different range re-scale
    im = im * (max - min) + min

    return im


def get_single_cell_coordinates(mask, cellid, expansion=10):
    # Focus on one cell at a time
    singlecell = mask == cellid

    # Get non-zero coordinates for x and y
    x,y = np.nonzero(singlecell)

    # Carefully expand the coordinates, we can not have coordinates less than 0
    # and we can not have coordinates greater than the size of the image
    x1 = x.min()-expansion if x.min()-expansion > 0 else 0
    x2 = x.max()+expansion if x.max()+expansion < mask.shape[0] else mask.shape[0]
    y1 = y.min()-expansion if y.min()-expansion > 0 else 0
    y2 = y.max()+expansion if y.max()+expansion < mask.shape[1] else mask.shape[1]

    # Return coordinates for the cell with expansion
    return x1, x2, y1, y2


def pad_image(im, size):
    # Create empy array of the desired size
    black = np.zeros(size)

    # Check if im size is smaller than desired size
    # TODO: Handle the case when is not
    if im.shape[0] < black.shape[0] and im.shape[1] < black.shape[1]:

        # Calculate the difference between im and size
        dif_x = black.shape[0]-im.shape[0]
        dif_y = black.shape[1]-im.shape[1]

        # Divide the difference between im and size by 2
        # Fill blank image with original image on the new coordinates.
        black[dif_x//2:dif_x//2+im.shape[0], dif_y//2:dif_y//2+im.shape[1]] = im

    return black


def main(args)

if __name__ == '__main__':
    main()