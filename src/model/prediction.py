import argparse
import time
import torch
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from augmentations import preprocess_test as preprocess


def get_args():
    # Script description
    description = """Predicts the probability of occurrence of micro-nuclei in the cropped images in the input folder."""

    # Add parser
    parser = argparse.ArgumentParser(description)

    # Tool Input
    input = parser.add_argument_group(title="Input")
    input.add_argument("-i", "--images", dest="images", action="store", required=True,
                       help="Pathway to image folder.")
    input.add_argument("-m", "--model", dest="model", action="store", required=True,
                       help="Pathway to prediction model.")
    input.add_argument("-d", "--device", dest="device", action="store", required=False, default="cpu",
                       help="Device to be used for training [default='cpu']")

    # Tool output
    output = parser.add_argument_group(title="Output")
    output.add_argument("-o", "--out", dest="out", action="store", required=True,
                        help="Path to the output data folder")

    # Parse arguments
    args = parser.parse_args()

    # Standardize paths
    args.images = Path(args.images).resolve()
    args.labels = Path(args.model).resolve()
    args.out = Path(args.out).resolve()

    return args


def main(args):
    # Get list of file names
    list_cropped_files = [path.name for path in args.images.iterdir()]

    # Load model and set to evaluation
    device = args.device
    print(f"Using device = {device}")
    net = torch.load(args.model, map_location=device)
    net.eval()

    # Iterate over files
    list_predictions = []
    for image in tqdm(list(args.images.iterdir())):
        img_pil = Image.open(image)
        img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
        y = net(img_tensor).cpu().detach().numpy()
        list_predictions.append(y[0][0])

    # Create dictionary with results
    dict_tmp = {
        "image": list_cropped_files,
        "score": list_predictions
    }

    # Create dataframe
    df_predictions = pd.DataFrame.from_dict(dict_tmp)

    # Get micronuclei counts
    df_predictions["micronuclei"] = df_predictions["score"].apply(lambda x: round(x) if x > 0.5 else 0)

    # Get dataset summary
    print("Calculating summary.")
    df_summary = df_predictions["micronuclei"].value_counts()

    total = df_summary.sum()
    total_micronuclei = sum(df_summary.index * df_summary.values)
    cells_with_micronuclei = df_summary[df_summary.index > 0].sum()
    cells_with_micronuclei_ratio = cells_with_micronuclei / total
    micronuclei_ratio = total_micronuclei / total

    # Add summary to dataframe
    df_summary["total_cells"] = total
    df_summary["total_micronuclei"] = total_micronuclei
    df_summary["cells_with_micronuclei"] = cells_with_micronuclei
    df_summary["cells_with_micronuclei_ratio"] = cells_with_micronuclei_ratio
    df_summary["micronuclei_ratio"] = micronuclei_ratio


    # Save output file
    print("Finished prediction. Saving output file.")
    args.out.mkdir(parents=True, exist_ok=True)
    df_predictions.to_csv(args.out.joinpath(f"{args.images.name}_predictions.csv"), index=False)
    df_summary.to_csv(args.out.joinpath(f"{args.images.name}_summary.csv"), index=True)


if __name__ == "__main__":
    # Read arguments from command line
    args = get_args()

    # Run script and calculate run time
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finish in {rt//60:.0f}m {rt%60:.0f}s")