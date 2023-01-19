import argparse
from pathlib import Path
from cellpose import models
from skimage import io
import time


# Parsing input file and parameters


def get_args():
    # Script description
    description = """Uses CellPose to segment nuclei and saves the segmentation mask as numpy array."""

    # Add parser
    parser = argparse.ArgumentParser(description)

    # Tool Input
    input = parser.add_argument_group(title="Input")
    input.add_argument("-i", "--image", dest="image", action="store", required=True, help="Pathway to input image.")
    input.add_argument("-c", "--channels", dest="channels", action="store", type=int, nargs="+", required=True,
                       help="List of channels used for segmentation of length 2. "
                            "2nd element  https://cellpose.readthedocs.io/en/latest/settings.htmlf")

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
    model = models.Cellpose(model_type='nuclei')

    # Read in data, it most be contained in a list object for evaluation
    img = [io.imread(args.image)]

    # Predict nuclei in image
    masks, flows, styles, diams = model.eval(img, channels=args.channels, diameter=None)

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
