
<img align="right" width="200" height="60" src= "images/logo_name.png">

# micronuclAI
**Automated assessment of Chromosomal Instability through quantification of micronuclei (MN) and Nuclear Buds (NBUDs).**

![screenshot](images/overview.png)

micronuclAI is an automated pipeline for reliable quantification of micronuclei of varying size, morphology and location from nuclei-stained images. The pipeline can be divided into 3 steps: 1) segmentation, 2) nuclei isolation, and 3) quantification. In brief, single-cell crops are extracted from high-resolution microscopy images with the help of segmentation masks, which are then used to train a convolutional neural network (CNN) to output the number of MN associated with each cell. 

## Getting Started

### Stream-lit App

A Quick implementation is available through 
[StreamLit](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2).

// TODO: Update the streamlit link once registered with the domain.

![screenshot](images/streamlit_app.png)

Inference of micronuclei can be achieved in small to medium sized example images that can be uploaded to the streamlit app. Image data is processed within a virtual machine (VM) on Heicloud, a local Cloud infrastructure provided by University Computing Center Heidelberg, and images are immediately deleted after micronuclei inference. Once micronuclei are inferred, results predictions as well as several plots describing the results are generated and presented to the user within the streamlit app which can also be downloaded as csv files. 

### Local Installation 

### Prerequisites


Two input files are required for micronuclAI
1. Nuclei-stained Image 
2. Associated Mask 

### Local Installation

A step by step series of examples that tell you how to get a development
environment running locally

Please ensure you have Python>=3.10 installed with 

```
python -version
```

Pull the repo into your local device

```cmd
git pull https://github.com/SchapiroLabor/micronuclAI.git
```
Install the required libraries
 ```
 pip install -r requirements.txt
 ```

## Usage

To use micronuclAI on your own images:

```
python src/model/prediction2.py -i <path/to/image> -m <path/to/mask> -mod <path/to/model> -o <path/to/output/folder>
```

A test dataset has been provided in the [test_data](test) folder with a [mask](test/test_mask.tiff) and its corresponding [image](test/test_image.tiff) file.

```
python src/model/prediction2.py -i /test/test_image.tiff -m /test/test_mask.tiff -mod /micronuclAI_model/micronuclai.pt -o /test/output
```

### Parameters and Arguments
| Parameter          | Short Form | Required | Default    | Type         | Description                                                                                       |
|--------------------|------------|----------|------------|--------------|---------------------------------------------------------------------------------------------------|
| `--image`          | `-i`       | Yes      | N/A        | String       | Pathway to image.                                                                                 |
| `--mask`           | `-m`       | Yes      | N/A        | String       | Pathway to mask.                                                                                  |
| `--model`          | `-mod`     | Yes      | N/A        | String       | Pathway to prediction model.                                                                      |
| `--out`            | `-o`       | Yes      | N/A        | String       | Path to the output data folder.                                                                   |
| `--size`           | `-s`       | No       | (256, 256) | List of int  | Size of images for training.                                                                      |
| `--resizing_factor`| `-rf`      | No       | 0.6        | Float        | Resizing factor for images.                                                                       |
| `--expansion`      | `-e`       | No       | 25         | Int          | Expansion factor for images.                                                                      |
| `--precision`      | `-p`       | No       | 32         | String       | Precision for training. Options: `["16-mixed", "bf16-mixed", "16-true", "bf16-true", "32", "64"]` |
| `--device`         | `-d`       | No       | "cpu"      | String       | Device to be used for training.                                                                   |
| `--batch_size`     | `-bs`      | No       | 32         | Int          | Batch size for training.                                                                          |
| `--workers`        | `-w`       | No       | 0          | Int          | Number of workers for training/testing.                                                           |


## Tutorial 
- [ ] TODO: A Jupiter notebook tutorial

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code
of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE) - see the [LICENSE](LICENSE) file and [ESSENTIAL_LICENSE_CONDITIONS.txt](ESSENTIAL_LICENSE_CONDITIONS.txt) for
details

