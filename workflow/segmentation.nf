#!/usr/bin/env nextflow
params.input = ""
params.device = ""
params.segmentation = ""
params.mpp = ""
params.batch_size = ""
params.cp_diameter = ""

// Set project and conda environment paths
conda = "$HOME/.conda/envs"
project = "$HOME/cin"

// Set script paths
script_cellpose = "$project/src/segmentation_cellpose.py"
script_stardist = "$project/src/segmentation_stardist.py"
script_deepcell = "$project/src/segmentation_deepcell.py"

// Set log message
log.info """\
     SEGMENTATION PIPELINE
     =========================
     input folder        = ${params.input}
     segmentation method = ${params.segmentation}
    """
    .stripIndent()

// Process for stardist segmentation
process SEGMENTATION_STARDIST{
    maxForks 1
    conda "${conda}/stardist"
    publishDir "${params.input}/segmentation/stardist", mode: "move"

    input:
    path (nuclear_image), stageAs: "input.ome.tif"

    output:
    path '*.tif'

    script:
    """
    python $script_stardist -i input.ome.tif -o .
    """
}

// Process for cellpose segmentation
process SEGMENTATION_CELLPOSE{
    maxForks 1
    conda "${conda}/cellpose"
    publishDir "${params.input}/segmentation/${params.segmentation}", mode: "move"

    input:
    path (nuclear_image), stageAs: "input.ome.tif"

    output:
    path '*.tif'

    script:
    def segmentation = params.segmentation.substring(params.segmentation.lastIndexOf("_") + 1)
    def model = "${segmentation}" ? "-m ${segmentation}" : ""
    def batch_size = "${params.batch_size}" ? "-bs ${params.batch_size}" : ""
    def cp_diameter = "${params.cp_diameter}" ? "-dm ${params.cp_diameter}" : ""

    """
    python $script_cellpose -i input.ome.tif -o . -d $params.device $model $cp_diameter $batch_size
    """
}

// Process for deepcell segmentation
process SEGMENTATION_DEEPCELL{
    maxForks 1
    conda "${conda}/deepcell"
    publishDir "${params.input}/segmentation/${params.segmentation}", mode: "move"

    input:
    path (nuclear_image), stageAs: "input.ome.tif"

    output:
    path '*.tif'

    script:
    def segmentation = params.segmentation.substring(params.segmentation.lastIndexOf("_") + 1)
    def model = "${segmentation}" ? "-m ${segmentation}" : ""
    def batch_size = "${params.batch_size}" ? "-bs ${params.batch_size}" : ""
    def mpp = "${params.mpp}" ? "-mpp ${params.mpp}" : ""

    """
    python $script_deepcell -i input.ome.tif -o . -mpp $params.mpp $model $batch_size
    """
}

workflow {
    // Getting input channel
    input_ch = Channel.fromPath("${params.input}/ometif/*")

    // Running segmentation process
    if (params.segmentation.contains("stardist")) {
        SEGMENTATION_STARDIST(input_ch)
        println "Using stardist segmentation"
    } else  if (params.segmentation.contains("cellpose")) {
        SEGMENTATION_CELLPOSE(input_ch)
        println "Using cellpose segmentation"
    } else if (params.segmentation.contains("deepcell")) {
        SEGMENTATION_DEEPCELL(input_ch)
        println "Using deepcell segmentation"
    } else {
        println "Please specify a segmentation method"
    }
}
