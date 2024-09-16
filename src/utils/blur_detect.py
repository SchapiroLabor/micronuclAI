## Description: This script detects blurred nuclei images by calculating the Laplacian variance 
## of images in a folder, provide some statistics and saves the results to a csv file. 
##
## Usage example: python blur_detect.py <path/to/folder/>
## Author: @arojhada
## Date: 2024-08-15

# Import the required libraries
import cv2
import os
import pandas as pd
import argparse
from tqdm import tqdm

def get_args():
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Calculate Laplacian variance of images in a folder')
    parser.add_argument('path', type=str, help='Path to the folder containing images')
    args = parser.parse_args()
    
    return args


def main(args):

    # Create an empty list to store the results
    results = []

    #Loop through the images in the folder and calculate the Laplacian variance
    for filename in tqdm(os.listdir(args.path)):
        if filename.endswith(".png"):
            img_path = args.path + filename
            img = cv2.imread(img_path)
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            results.append([filename, laplacian_var])
        else:
            continue

    #save the results to a csv file
    results = pd.DataFrame(results, columns=['filename', 'laplacian_var'])
    results.to_csv('output.csv', index=False)

    # Display the results
    thresh = 5.5   # threshold for blurry images
    num_blurry = results[results['laplacian_var'] < thresh].shape[0]
    blurry_percent = num_blurry/results.shape[0] * 100
    print("-------------Results:------------")
    print("Number of total images:       ", results.shape[0])
    print("Number of blurry images (%):  ", num_blurry, "(", "%.2f" % blurry_percent, "%)")
    print("Number of non-blurry images:  ", results.shape[0] - num_blurry)


if __name__ == '__main__':
    args = get_args()
    main(args)