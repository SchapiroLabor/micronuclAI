from pathlib import Path
from skimage import io
from skimage import exposure
import argparse
import numpy as np
import time


def get_args():
    # Script description
    description = """Preprocesses images with 99.9 Percentile Scaling and CLAHE."""

    # Add parser
    parser = argparse.ArgumentParser(description)
    parser.add_argument("-i", "--image", dest="image", action="store", required=True,
                        help="Pathway to input image.")
    parser.add_argument("-o", "--out", dest="out", action="store", required=True,
                        help="Path to the output data folder.")

    # Parse arguments
    args = parser.parse_args()

    # Standardize paths
    args.image = Path(args.image).resolve()
    args.out = Path(args.out).resolve()

    return args


def percentile_scaling(img, percentile):
    """
    Rescales the images to (0,1) and crops at percentile of initial maximal intensity.
    """
    top_percentile = np.percentile(img, percentile)
    img_rescaled = exposure.rescale_intensity(img, in_range=(0, top_percentile), out_range=(0, 1))

    return img_rescaled


def main(args):
    # Read image
    img = io.imread(args.image)
    print(f"Preprocessing {args.image.name}")

    # 99.9 percentile scaling and clahe only on the nuclear channel
    img_rescaled = percentile_scaling(img[:, :, 1], 99.9)
    img_equalized = exposure.equalize_adapthist(img_rescaled)

    # Save file, create output folder if it does not  exist
    Path(args.out).mkdir(parents=True, exist_ok=True)
    io.imsave(args.out.joinpath(f"{args.image.name}"), img_equalized)


if __name__ == "__main__":
    # Read arguments from command line
    args = get_args()

    # Run script
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finish in {rt//60:.0f}m {rt%60:.0f}s")