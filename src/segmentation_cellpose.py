# Built-in libraries
import argparse
import time
from pathlib import Path

# External librarires
import torch
from skimage import io
from cellpose import models
from PIL import Image


# Parsing input file and parameters
def get_args():
    # Script description
    description = """Uses CellPose to segment nuclei and saves the segmentation mask as numpy array."""

    # Add parser
    parser = argparse.ArgumentParser(description)

    # Tool Input
    input = parser.add_argument_group(title="Input")
    input.add_argument("-i", "--image", dest="image", action="store", required=True, help="Pathway to input image.")
    input.add_argument("-d", "--device", dest="device", action="store", default=None, required=False,
                       help="Use gpu for inference acceleration.")
    input.add_argument("-m", "--model", dest="model", action="store", required=False, default="nuclei",
                       choices=["nuclei", "cyto", "cyto2"],
                       help="Model to be used for segmentation [default='nuclei'].")
    input.add_argument("-dm", "--diameter", dest="diameter", action="store", required=False, default=None, type=float,
                       help="Diameter of the nuclei [default=None].")

    # Cellpose options
    optional = parser.add_argument_group(title="Cellpose options")
    optional.add_argument("-ft", "--flow-threshold", dest="flow_threshold", action="store", required=False, default=0.4,
                          type=float, help="Flow threshold to be used for calculations. [default=0.4]")
    optional.add_argument("-bs", "--batch-size", dest="batch_size", action="store", required=False, default=8,
                          type=int, help="Batch size [default=8]")

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
    print(f"Loading Model with device = {args.device}")
    print(f"Loading Model             = {args.model}")
    model = models.Cellpose(model_type=args.model, device=torch.device(args.device), gpu=True)

    # Read in data, it most be contained in a list object for evaluation
    print(f"Reading image from        = {args.image}")
    img = io.imread(args.image)

    print(f"Image with shape          = {img.shape}")
    img = [img]

    # Predict nuclei in image
    print("Predicting")
    masks, flows, styles, diams = model.eval(img, 
                                            channels=[0, 0], 
                                            diameter=args.diameter, 
                                            flow_threshold=args.flow_threshold,
                                            batch_size=args.batch_size)

    print(f"Predicted mask with shape = {masks[0].shape}")
    print(f"Predicted diams used      = {diams}")
    print(f"Predicted masks           = {masks[0].max()}")

    # Create output folder if it does not exist
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Save mask file in tif format
    print(f"Saving mask to             = {args.out}")
    io.imsave(args.out.joinpath(f"{args.image.name.split('.')[0]}.tif"), masks[0])


if __name__ == "__main__":
    # Read arguments from command line
    args = get_args()

    # Run script
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finish in {rt//60:.0f}m {rt%60:.0f}s")
