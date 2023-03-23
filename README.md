# micronuclAI

Chromosomal Instability, micronuclei automatic detection.

## Pipeline overview
We perform a series of steps for training the MicronucleAI model.
1. Pre-Processing
2. Segmentation
3. Nuclei Isolation / Nuclei Cropping
4. Labeling
5. Training
6. Prediction

If you only want to use the predictive model you must follow the following steps.
1. Pre-Processing
2. Segmentation
3. Nuclei Isolation / Nuclei Cropping
4. Prediction

We will describe the steps in detail for the training procedure.

# Training of the model

## Raw data

The dataset used for training consist of 216 20x zoom `.czi` image files obtained from 8 different plates (see table).

| Plate           | # of Files |
|-----------------|------------|
| A375_dnMCAK-dox | 25         |
| A375_dnMCAK+dox | 27         |
| A375_kifa-dox   | 24         |
| A375_kifa+dox   | 33         |
| A375_kifc-dox   | 27         |
| A375_kifc+dox   | 24         |
| A375-dox        | 26         |
| A375+dox        | 30         |
| Total           | 216        |

Each `.czi` file has 3 channels: Cytoskeleton, DAPI and auto-florescence channel.

| Channel Number | Marker         |
|----------------|----------------|
| 0              | Cytoskeleton   |
| 1              | DAPI           |
| 2              | Autoflouresnce |

## Pre-Processing

In this step we re-formated the `.czi` formated image files to `.ome.tif` and select only the DAPI channel for further nuclear segmentation. 
To perform the convertion we use the script `src/czi2ometif.py`. This script makes use of the AICSImageio library to handle the conversion.

Example for one image:
```
python czi2ometif.py -i path/to/image_1.czi -o path/to/converted -c 1
```

A way to do this in a single command is to use Nextflow for processing using the following command.
```
nextflow run workflow/czi2ometif.nf --input "/Users/miguelibarra/PycharmProjects/cin/data/training/raw/*.czi" --channel 1 --output /Users/miguelibarra/PycharmProjects/cin/data/training/ometif -with-conda true
```

Processing time of 4m 19s on M1 Mac 32 Gb RAM.

## Segmentation 

Segmentation will aid us identifying and labeling the nuclei in our image. For the segmentation part of the pipeline you can use any segmentation method you find useful. In this case we tested CellPose[reference] and Mesmer[reference]. 

- MESMER segmentation using the MCMICRO (version  f2c2433) [reference] implementation with the paremeters `mcm_parameters/parameters_original_training.yml`  
```
nextflow run labsyspharm/mcmicro --in data/original/mcm_original/ --params mcm_parameters/parameters_original_mesmer.yml -profile singularity -c ~/exemplar.config 
```
- Cellpose segmentation was performed using the pre-trained Cellpose model (version 2.2) [reference] and with the script `src/cellpose_segmentation.py`
```
nextflow run workflow/mask_cellpose.nf  --input --output
```
 
## Nuclei isolation

To train a CNN/DNN is necessary to have a standardized data format. For this instance, we are interested into training an EfficientNet B-0 binary classifier using an image size of 256 x 256 pixels per image. Each image containing (ideally) a single nuclei.  

The process we follow for extracting single nucleus images goes as follows:

1. Get the corresponding image and mask.
2. Get the bounding box of each identify nuclei.
3. Calculate scale factor to match the desired cell/image ratio.
4. Expand the field of view by a given factor (in this case 20 pixels on each side)
6. Remove the cells that are smaller than a certain threshold (in this case 140 pixels on either height or width).
7. Re-scale the image to the desired cell/image ration (in this case 0.7)
8. Pad and crop the image as necessary to the desired size (256 x 256)
9. Re-scale the brightness intensity values
10. Save as an 8-bit `.png` image.

With this processing pipeline we are able to retrieve 2410 single nuclei.

```
nextflow run workflow/isolate_cells.nf --masks "/Users/miguelibarra/PycharmProjects/cin/data/training/segmentation/mesmer_whole/*" --images "/Users/miguelibarra/PycharmProjects/cin/data/training/ometif" --output "/Users/miguelibarra/PycharmProjects/cin/data/training/isonuc/mesmer_wc_" --fv 20 --ds 0.7 --xms 140 --yms 140  -with-conda true
```

## Labeling

We labeled the 2,642 single nuclear images by using the labeling tool.
1. According to Miguel's labeling: 

| # micronuclei | frequency |
|---------------|-----------|
| 0             | 2148      |
| 1             | 419       |
| 2             | 53        |
| 3             | 15        |
| 4             | 5         |
| 5             | 2         |
| 6             | 1         |

## Training



## Prediction

