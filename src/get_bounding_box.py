# Built in libraries
import time
import argparse
from pathlib import Path
from skimage import exposure, transform
from aicsimageio import AICSImage

# External libraries
import numpy as np
from skimage import io


def get_options():
    # Script description
    description="""Converts a nuclear into images for each single nuclei"""

    # Add parser
    parser = argparse.ArgumentParser(description)

    # Tool Input
    input = parser.add_argument_group(title="Input")
    input.add_argument("-i", "--image", dest="image", action="store", required=True, help="Pathway to input image.")
    input.add_argument("-m", "--mask", dest="mask", action="store", required=True, help="Pathway to mask segmentation.")
    input.add_argument("-c", "--channel", dest="channel", action="store", required=False, help="DAPI channel.")

    # Tool options
    options = parser.add_argument_group(title="Options")
    options.add_argument("-fv", "--field-view", dest="field_view", action="store", required=False, default="50", type=int,
                         help="Number of pixels to expand each cell bounding box, field of view of each cell "
                              "[default = 50]")
    options.add_argument("-fs", "--final-size", dest="fsize", action="store", required=False, default=256, type=int,
                         help="Final desired size of the isolated nuclei, they will be squared. [default=256].")
    options.add_argument("-re", "--remove-edge", dest="remove_edge", action="store_true", required=False, default=False,
                         help="Flag to remove cells that lay on the edge of the image.")
    options.add_argument("-s", "--scaling-factor", dest="scaling_factor", action="store", required=False, default=None,
                         type=float, help="Scaling factor values < 1 make image smaller, >0 make it big [default=None]")
    options.add_argument("-xms", "--x-min-size", dest="x_min_size", action="store", required=False, default=0, type=int,
                         help="Limit for the minimum size required for the x axis in an image.")
    options.add_argument("-yms", "--y-min-size", dest="y_min_size", action="store", required=False, default=0, type=int,
                         help="Limit for the minimum size required for the y axis in an image.")
    # Tool output
    output = parser.add_argument_group(title="Output")
    output.add_argument("-o", "--out", dest="out", action="store", required=True, help="Path to the output data folder")

    # Parse arguments
    args = parser.parse_args()

    # Stan dardize paths
    args.image = Path(args.image).resolve()
    args.mask = Path(args.mask).resolve()
    args.out = Path(args.out).resolve()

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


def get_single_cell_coordinates(mask, cellid, expansion):
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
    x, y = np.nonzero(singlecell)

    # Carefully expand the coordinates, we can not have coordinates less than 0
    # and we can not have coordinates greater than the size of the image
    x1 = x.min()-expansion if x.min()-expansion > 0 else 0
    x2 = x.max()+expansion if x.max()+expansion < mask.shape[0] else mask.shape[0]
    y1 = y.min()-expansion if y.min()-expansion > 0 else 0
    y2 = y.max()+expansion if y.max()+expansion < mask.shape[1] else mask.shape[1]

    # Return coordinates for the cell with expansion
    return np.array([x1, x2, y1, y2])


def pad_image(im, size):
    """
    Inserts an image at the center of a black canvas.

    :param im: image to insert.
    :param size: Canvas size.
    :return: Canvas with inserted image.
    """
    # Create empy array of the desired size
    black = np.zeros(size)

    # Check if im size is smaller than desired size, prefer to use resize to handle cases where this is not true
    if im.shape[0] < black.shape[0] and im.shape[1] < black.shape[1]:

        # Calculate the difference between im and size
        dif_x = black.shape[0]-im.shape[0]
        dif_y = black.shape[1]-im.shape[1]

        # Divide the difference between im and size by 2
        # Fill blank image with original image on the new coordinates.
        black[dif_x//2:dif_x//2+im.shape[0], dif_y//2:dif_y//2+im.shape[1]] = im

    return black


def resize(im, size):
    """
    Makes an image square to a desire size without changing the ratio

    :param im: Image to pad
    :param size: Final desired size
    :return: Image of the desired sized, with added padding where needed
    """
    ox = im.shape[0]
    oy = im.shape[1]

    # Get the difference between the current size and the desired size
    dif_x = size[0] - im.shape[0]
    dif_y = size[1] - im.shape[1]

    # Getting the difference for each size of the image
    dif_x1 = dif_x//2
    dif_x2 = dif_x//2 + dif_x % 2
    dif_y1 = dif_y//2
    dif_y2 = dif_y//2 + dif_y % 2
    difs = np.array([dif_x1, dif_x2, dif_y1, dif_y2])

    # Get crop differences
    dif_crop = np.where(difs < 0, -difs, 0)

    # Get pad differences
    dif_pad = np.where(difs >= 0, difs, 0)

    # Remove pixels from image if difference is negative
    im = im[0+dif_crop[0]:ox-dif_crop[1], 0+dif_crop[2]:oy-dif_crop[3]]

    # Assuming the image is smaller than the desired size
    im = np.pad(im, [(dif_pad[0], dif_pad[1]), (dif_pad[2], dif_pad[3])], mode="constant")

    return im


def main(args):
    # Parameters
    print(f"Removing cells in edges = {args.remove_edge}")
    print(f"Removing X minimum size = {args.x_min_size}")
    print(f"Removing Y minimum size = {args.y_min_size}")
    print(f"Using scalling factor   = {args.scaling_factor}")

    # Load image, and mask
    image = AICSImage(args.image, C=args.channel).get_image_data("YX")
    mask = AICSImage(args.mask).get_image_data("YX")
    print(f"Image shape = {image.shape}")
    print(f"Mask shape  = {mask.shape}")

    # Iterate over each cell in the mask and get single cell bounding boxes sc_bb
    # CellIDs start from 1 that's why I used range from 1 to n+1
    # Doing this cell by cell since it can be easily parallelized
    print("Getting bounding boxes")
    mask_sc_bb = [get_single_cell_coordinates(mask, i, expansion=args.field_view) for i in range(1, mask.max() + 1)]

    # We now iterate over each predicted bounding box
    print("Expanding bounding box and save")
    for cellid, coord in enumerate(mask_sc_bb):
        # Get the coordinates of the bounding box for each cell mask
        x1, x2, y1, y2 = coord

        # Remove cells from the edge
        if args.remove_edge and (any(coord == 0) or any(coord == mask.shape[0]) or any(coord == mask.shape[1])):
            print(f"{args.image.name} cell {cellid} removed: edge")
            continue

        # Remove cells if they re bellow a certain threshold
        if (x2-x1) < args.x_min_size and (y2-y1) < args.y_min_size:
            print(f"{args.image.name} cell {cellid} removed: small")
            continue

        # Get single cell image data
        sc = image[x1:x2, y1:y2]

        # Rescale images
        if args.scaling_factor is not None:
            sc = transform.rescale(sc, args.scaling_factor)

        # Pad/Crop image tp the desired size
        sc = resize(sc, size=(args.fsize, args.fsize))

        # Scale image to 8bit for jpeg transform
        sc = exposure.rescale_intensity(sc, out_range=(0, 255)).astype(np.uint8)

        # Save SC image
        args.out.mkdir(parents=True, exist_ok=True)
        io.imsave(str(args.out.joinpath(f"{args.image.name.split('.')[0]}_{cellid}.png")), sc)


if __name__ == '__main__':
    # Get arguments
    args = get_options()

    # Run script
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finish in {rt//60:.0f}m {rt%60:.0f}s")