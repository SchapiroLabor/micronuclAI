# Built-in libraries
import argparse
import time
from pathlib import Path

# External librarires
import torch
from skimage import io
from cellpose import models
from PIL import Image
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter


# Parsing input file and parameters
def get_args():
    # Script description
    description = """Uses CellPose to segment nuclei and saves the segmentation mask as numpy array."""

    # Add parser
    parser = argparse.ArgumentParser(description)

    # Tool Input
    input = parser.add_argument_group(title="Input")
    input.add_argument("-i", "--image", dest="image", action="store", required=True, help="Pathway to input image.")
    input.add_argument("-c", "--channel", dest="channel", action="store", type=int, required=False, default=None,
                       help="Channel to be used in the original image.")
    input.add_argument("-g", "--gpu", dest="gpu", action="store_true", default=False, required=False,
                       help="Use gpu for inference acceleration.")

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
    # Load model
    print(f"Loading Model with gpu: {args.gpu}")
    model = models.Cellpose(model_type='nuclei', gpu=args.gpu)

    # Read in data, it most be contained in a list object for evaluation
    print("Reading image")
    img = io.imread(args.image)

    print(f"Image with shape: {img.shape}")
    img = [img]

    # Predict nuclei in image
    print("Predicting")
    masks, flows, styles, diams = model.eval(img, channels=[0, 2], diameter=None)

    # Create output folder if it does not exist
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Save mask file in tif format
    io.imsave(args.out.joinpath(f"{args.image.name.split('.')[0]}.tif"), masks[0])


if __name__ == "__main__":
    # Read arguments from command line
    args = get_args()

    # Run script
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finish in {rt//60:.0f}m {rt%60:.0f}s")
