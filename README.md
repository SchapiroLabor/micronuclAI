# micronuclAI

Chromosomal Instability, micronuclei automatic detection.

# Training of the model

## Raw data

Original raw data consist of 217 20x zoom `.czi` image files belonging to 8 different plates.  

| Plate           | # of Files |
|-----------------|------------|
| A375_dnMCAK-dox | 26         |
| A375_dnMCAK+dox | 27         |
| A375_kifa-dox   | 24         |
| A375_kifa+dox   | 33         |
| A375_kifc-dox   | 27         |
| A375_kifc+dox   | 24         |
| A375-dox        | 26         |
| A375+dox        | 30         |
| Total           | 217        |

Each file is contained 3 channels: Cytoskeleton, DAPI and autoflorescence channel.

| Channel Number | Marker         |
|----------------|----------------|
| 0              | Cytoskeleton   |
| 1              | DAPI           |
| 2              | Autoflouresnce |

## Pipeline overview
We perform a series of steps for training the MicronucleAI model.
1. Pre-Processing
2. Segmentation
3. Nuclei Isolation / Nuclei Cropping
4. Labeling
5. Training
6. Prediction

If you only want to use the predictive model you must folow the following steps.
1. Pre-Processing
2. Segmentation
3. Nuclei Isolation / Nuclei Cropping
4. Prediction

We will describe the steps in detail for the training procedure.

## Pre-Processing

The goal of pre-processing our dataset is to convert multiple format datasets to `.ome.tif` ones. We also want to work only with the nuclear channel (DAPI in this case).
For the pre-processing we converted the 216 `.czi` into `.ome.tif` files using `src/czi2ometif.py`
A way to do this in a single command is to use Nextflow for processing using the following command.  


```
nextflow run workflow/czi2ometif.nf --input "/Users/miguelibarra/PycharmProjects/cin/data/training/raw/*.czi" --output /Users/miguelibarra/PycharmProjects/cin/data/training/ometif -with-conda true
```
Processing time of 4m 19s on M1 Mac 32 Gb.  

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


## Training


## Prediction

