# Import local librarires
from pathlib import Path
import time
import argparse

# Import external libraries
from skimage import io
from skimage.color import rgb2gray
from PIL import Image
# from aicsimageio import AICSImage
Image.MAX_IMAGE_PIXELS = None
from aicsimageio import AICSImage


# Argument parser
def get_args():
    # Script description
    description = """Convert jpeg images to tif images."""

    # Add parser
    parser = argparse.ArgumentParser(description)
    parser.add_argument("-i", "--image", dest="image", action="store", required=True,
                        help="Pathway to input image.")
    parser.add_argument("-c", "--channel", dest="channel", action="store", required=False, type=int, default=None,
                        help="Channel to be used, if not channel provided all the image is saved.")
    parser.add_argument("-o", "--out", dest="out", action="store", required=True,
                        help="Path to the output data folder.")

    # Parse arguments
    args = parser.parse_args()

    # Standardize paths
    args.image = Path(args.image).resolve()
    args.out = Path(args.out).resolve()

    return args


def main(args):
    # Read image
    img = io.imread(args.image)
    print(f"Image shape = {img.shape}")


    # Deal with the channels
    if args.channel is not None:
        img = img[:, :, args.channel]
    else:
        img = rgb2gray(img)

    # Save image
    args.out.mkdir(parents=True, exist_ok=True)
    # io.imsave(args.out.joinpath(f"{args.image.stem}.ome.tif"), img)
    AICSImage(img).save(args.out.joinpath(f"{args.image.stem}.tif"))


if __name__ == "__main__":
    # Read arguments from command line
    args = get_args()

    # Run script
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finish in {rt//60:.0f}m {rt%60:.0f}s")