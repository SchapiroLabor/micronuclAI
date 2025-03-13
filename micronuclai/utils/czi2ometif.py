# Import local libraries
import time
from pathlib import Path
import argparse
from argparse import ArgumentParser as AP

# Import external libraries
from aicsimageio import AICSImage

from aicsimageio.writers import OmeTiffWriter


def get_args():
    # Script description
    description = "Converts a .czi file to an .ome.tiff file using aicsimageio package"

    # Initialize parser
    parser = AP(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Parser sections
    tool = parser.add_argument_group(title="Required Input", description="Required tool input.")
    tool.add_argument("-i", "--image", dest="image", action="store", required=True, type=str,
                      help="Path to input .czi image.")
    tool.add_argument("-c", "--channel", dest="channel", action="store", required=False, type=int, default=None,
                        help="Channel to be used, if not channel provided all the image is saved.")
    tool.add_argument("-s", "--series", dest="series", action="store", required=False, type=int, default=None,
                        help="Series to be used, if not series provided the first series is used.")

    out = parser.add_argument_group(title="Output", description="Output files.")
    out.add_argument("-o", "--output", dest="output", action="store", required=True, type=str,
                     help="Path to folder where to output the .ome.tiff image.")

    arg = parser.parse_args()

    # Standardize paths
    arg.image = Path(arg.image).resolve()
    arg.output = Path(arg.output).resolve()

    return arg


def main(args):
    # Read .czi image
    img = AICSImage(args.image)
    print(f"Image shape      = {img.shape}")
    print(f"Image dimensions = {img.dims}")

    # Deal with channels
    if args.channel is not None:
        img = img.get_image_data("YX", C=args.channel)
        print(f"Image shape      = {img.shape}")
        img = AICSImage(img)

    if args.series is not None:
        img = img.get_image_data("YX", S=args.series)
        print(f"Image shape      = {img.shape}")
        img = AICSImage(img)

    # Save .ome.tif image
    print("Saving output image")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.output)


if __name__ == '__main__':
    # Read arguments from command line
    args = get_args()

    # Run script
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finish in {rt//60:.0f}m {rt%60:.0f}s")

