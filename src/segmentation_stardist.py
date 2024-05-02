# Built-in libraries
import argparse
import time
from pathlib import Path

# External libraries
from skimage.io import imread, imsave
from stardist.models import StarDist2D
from csbdeep.utils import normalize


# Parsing input file and parameters
def get_args():
    # Script description
    description = """Uses Stardist to segment nuclei and saves the segmentation mask as numpy array."""

    # Add parser
    parser = argparse.ArgumentParser(description)

    # Tool Input
    input = parser.add_argument_group(title = "Input")
    input.add_argument("-i", "--image", dest="image",
                       action="store", required=True,
                       help="Pathway to input image.")
    input.add_argument("-m", "--model", dest="model",
                       action="store", required=False, default="2D_versatile_fluo",
                       help="Model to be used for segmentation [default='2D_versatile_fluo'].")
    parser.add_argument("-pnorm", "--pnorm", dest="pnorm",
                        type=float, nargs=2, default=[1, 99.8],
                        help="pmin/pmax to use for normalization")

    # Tool output
    output = parser.add_argument_group(title = "Output")
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
    print(f"Loading Model      = {args.model}")
    model = StarDist2D.from_pretrained(args.model)

    # Read in data, it most be contained in a list object for evaluation
    print(f"Reading image")
    img = imread(args.image)
    print(f"Image has shape    = {img.shape}")

    # Normalize image
    img = normalize(img, *args.pnorm)

    # Predict nuclei
    print("Predicting")
    labels, _ = model.predict_instances(img)
    print(f"Mask shape         = {labels.shape}")
    print(f"Mask unique values = {len(set(labels.flatten()))}")
    print(f"Mask data type     = {labels.dtype}")

    # Create output folder if it does not exist
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving mask to     = {args.out}")

    # Save mask file in tif format
    imsave(args.out, labels)


if __name__ == "__main__":
    # Read arguments from command line
    args = get_args()

    # Run script
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finish in {rt//60:.0f}m {rt%60:.0f}s")
