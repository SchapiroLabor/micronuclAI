# Import libraries
import time
import argparse
from pathlib import Path

# Import external libraries
import numpy as np
from tqdm import tqdm
from mask2bbox import BBoxes
import matplotlib.pyplot as plt


def get_args():
    # Script description
    description = 'Extract single nuclei from a mask'

    # Add parser
    parser = argparse.ArgumentParser(description=description)

    # Tool input arguments
    inputs = parser.add_argument_group('Inputs')
    inputs.add_argument('-m', '--mask', dest='mask', type=str, required=True,
                        help='Path to the mask')
    inputs.add_argument('-i', '--image', dest='image', type=str, required=True,
                        help='Path to the image')

    # Tool optional arguments
    options = parser.add_argument_group('Options')
    options.add_argument('-s', '--size', dest='size', type=int, default=256, nargs="+",
                         help='Size of the extracted nuclei')
    options.add_argument('-e', '--expansion', dest='expansion', type=int, default=0,
                         help='Expansion of the bounding box in pixels')
    options.add_argument('-rf', '--resizing_factor', dest='resizing_factor', type=float, default=1.0,
                         help='Scale factor of the bounding box')
    options.add_argument('-fb', '--filter_bboxes', dest='filter_bboxes', action='store_true',
                         help='Filter the bounding boxes by area')

    # Tool output arguments
    outputs = parser.add_argument_group('Outputs')
    outputs.add_argument('-o', '--output', dest='output', type=str, required=True,
                         help='Path to the output directory')

    # Parse arguments
    arg = parser.parse_args()

    # Standardize paths
    arg.mask = Path(arg.mask).resolve()
    arg.image = Path(arg.image).resolve()
    arg.output = Path(arg.output).resolve()

    return arg


def main(args):
    # Crate BBoxes object
    print(f"Creating BBoxes object from {args.mask}")
    bboxes = BBoxes.from_mask(args.mask)
    print(f"Creating BBoxes object from {args.image}")
    bboxes.image = args.image
    print(f"Found {len(bboxes)} bounding boxes")

    # Get resizing factor using the original bounding boxes
    rf = bboxes.calculate_resizing_factor(args.resizing_factor, args.size)

    # Filter the bounding boxes by area if requested
    if args.filter_bboxes:
        # Get sides of the bounding boxes
        sides = bboxes.get("sides")

        # Compute the threshold for the 90th percentile
        print(f"Computing threshold for the 90th percentile of the bounding boxes' sides")
        threshold = np.percentile(sides[:,1], 10)

        # Create a boolean mask to filter the array
        print(f"Filtering bounding boxes with area greater than {threshold}")
        bboxes = bboxes.filter("sides", np.greater_equal, (threshold,threshold))


    # Expand the bounding boxes
    print(f"Expanding bounding boxes by {args.expansion} pixels")
    bboxes = bboxes.expand(args.expansion)

    # Extract nuclei from the filtered array
    print(f"Extracting {len(bboxes)} nuclei from the filtered array")
    bboxes.extract(rf[bboxes.idx()], args.size, args.output)


if __name__ == '__main__':
    # Get arguments
    arg = get_args()

    # Run script
    st = time.time()
    main(arg)
    rt = time.time() - st
    print(f"Script finish in {rt // 60:.0f}m {rt % 60:.0f}s")
