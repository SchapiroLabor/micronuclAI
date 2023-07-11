# Built-in libraries
import argparse
import time
from pathlib import Path

# External librarires
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from deepcell.applications import NuclearSegmentation, Mesmer
import numpy as np


# Parsing input file and parameters
def get_args():
    # Script description
    description = """Uses CellPose to segment nuclei and saves the segmentation mask as numpy array."""

    # Add parser
    parser = argparse.ArgumentParser(description)

    # Tool Input
    input = parser.add_argument_group(title="Input")
    input.add_argument("-i", "--image", dest="image", action="store", required=True,
                       help="Pathway to input image.")
    input.add_argument("-m", "--model", dest="model", action="store", required=False, default="nuclear",
                       choices=["nuclear", "mesmer"],
                       help="Model to be used for segmentation [default='nuclear'].")

    optional = parser.add_argument_group(title="Optional arguments")
    optional.add_argument("-mpp", "--mpp", dest="mpp", action="store", required=False, default=0.65, type=float,
                          help="Microns per pixel of the image [default=0.65].")
    optional.add_argument("-c", "--compartment", dest="compartment", action="store", required=False, default="nuclear",
                          help="Compartment to be segmented [default='nuclear'].")

    # Tool output
    output = parser.add_argument_group(title="Output")
    output.add_argument("-o", "--out", dest="out", action="store", required=True,
                        help="Path to the output data folder.")

    # Parse arguments
    args = parser.parse_args()

    # Standardize paths
    args.image = Path(args.image).resolve()
    args.out = Path(args.out).resolve()

    return args


def main(args):
    # Read data
    print(f"Reading image from    = {args.image}")
    img = imread(args.image)
    print(f"Image with shape      = {img.shape}")

    # Expand image dimensions to rank 4
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    # Create the application
    print(f"Predicting with model = {args.model}")
    print(f"Image mpp             = {args.mpp}")

    if args.model == "nuclear":
        app = NuclearSegmentation()
        labeled_image = app.predict(img, image_mpp=args.mpp)
    elif args.model == "mesmer":
        app = Mesmer()
        labeled_image = app.predict(img, image_mpp=args.mpp, compartment=args.compartment)

    # Save the label image
    mask = labeled_image[0, :, :, 0]
    args.out.mkdir(parents=True, exist_ok=True)
    imsave(args.out.joinpath(f"{args.image.name.split('.')[0]}.tif"), mask)


if __name__ == "__main__":
    # Read arguments from command line
    args = get_args()

    # Run script
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finish in {rt//60:.0f}m {rt%60:.0f}s")
