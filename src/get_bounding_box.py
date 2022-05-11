# Built in libraries
import os
import sys
import time
import argparse
from pathlib import Path
import concurrent.futures
from os.path import abspath, join, basename

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
    """
    Min-Max scaling to a given image (one channel only).
    It's possible to specify the desired range (default 0 to 1).

    :param im: Single channel image/array (Numpy, opencv, scikit-image compatible)
    :param minmaxrange: New range of values, by default 0 to 1
    :return: Scaled image in the specified range
    """
    # Unpack value range
    min, max = minmaxrange

    # Normal scaling (0 - 1)
    im = (im - im.min()) / (im.max() - im.min())

    # In case of different range re-scale
    im = im * (max - min) + min

    return im


def get_single_cell_coordinates(mask, cellid, expansion=10):
    """
    This function gets the mask bounding box from a specific cellID.
    The field of view can be increased if required with the expansion parameters.

    :param mask: tif image with nuclear/cell segmentation
    :param cellid: ID of the cell you want the bounding box.
    :param expansion: Amount of pixels added to the bounding box to each side.
    :return: Coordinates on mask file of the single cell/single nuclei bounding box
    """

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
    """
    Inserts an image at the center of a black canvas.

    :param im: image to insert.
    :param size: Canvas size.
    :return: Canvas with inserted image.
    """
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


def main(args):
    # Load image, and mask
    image = io.imread(args.image)
    mask = io.imread(args.mask)

    # Iterate over each cell in the mask and get single cell bounding boxes sc_bb
    # CellIDs start from 1 that's why I used range from 1 to n+1
    # Doing this cell by cell since it can be easily parallelized
    mask_sc_bb = [get_single_cell_coordinates(mask, i, expansion=50) for i in range(1, mask.max() + 1)]

    # We now iterate over each predicted bounding box
    for cellid, coord in enumerate(mask_sc_bb):
        x1, x2, y1, y2 = coord

        # Get single cell
        sc = image[x1:x2, y1:y2]

        # Scale image
        sc = im_min_max_scaler(sc, minmaxrange=(0, 255))

        # Pad image
        sc = pad_image(sc, (512, 512))

        # Save SC image
        cv2.imwrite(join(args.out, f"{basename(args.image)}_{cellid}.png"), sc)


if __name__ == '__main__':
    main()