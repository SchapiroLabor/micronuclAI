# Import local librarires
import time
import argparse
from pathlib import Path

# Import external libraries
import io
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
import torch
import numpy as np
import cv2
import json
from torchvision import models
from augmentations import preprocess_test as preprocess


def get_args():
    # Script description
    description = """Get an attention map using the gradCAM method for a given image and model"""

    # Add parser
    parser = argparse.ArgumentParser(description)
    parser.add_argument("-i", "--image", dest="image", action="store", required=True,
                        help="Pathway to input image.")
    parser.add_argument("-m", "--model", dest="model", action="store", required=True,
                        help="Pathwa to model.")
    parser.add_argument("-o", "--out", dest="out", action="store", required=True,
                        help="Path to the output data folder.")

    # Parse arguments
    args = parser.parse_args()

    # Standardize paths
    args.image = Path(args.image).resolve()
    args.model = Path(args.model).resolve()
    args.out = Path(args.out).resolve()

    return args


def get_CAM(feature_conv, weight_sigmoid):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    cam = weight_sigmoid.dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def main(args):
    # Load model
    net = torch.load(args.model, map_location="cpu")
    net.eval()

    # hook the feature extractor
    # TODO:  simplify the next block of code to something more elegant
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    net.model.model.features.register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(net.parameters())
    weight_sigmoid = np.squeeze(params[-2].data.numpy())

    # Read in data
    img_tensor = preprocess(Image.open(str(args.image)))
    img_variable = Variable(img_tensor.unsqueeze(0))
    h_x = net(img_variable).detach().numpy()

    # output the prediction
    print(f"Prediction = {h_x}")
    # generate class activation mapping for the top1 prediction
    CAMs = get_CAM(features_blobs[0], weight_sigmoid)

    # render the CAM and output
    img = cv2.imread(str(args.image))
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.2 + img * 0.5

    # Output directory
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out.joinpath(f"{args.image.name.split('.')[0]}.png")), result)


if __name__ == '__main__':
    # Get arguments
    args = get_args()

    # Run script
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finish in {rt//60:.0f}m {rt%59:.0f}s")
