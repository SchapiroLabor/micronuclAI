#!/usr/bin/env nextflow
params.input = ""
params.device = ""
params.segmentation = ""
params.mpp = ""
params.batch_size = ""
params.cp_diameter = ""

// Set project and conda environment paths
params.c = ""
params.s = ""

params.model = ""
params.size = "256 256"
params.rf = "0.6"
params.e = "30"


// Set project and conda environment paths
conda = "$HOME/.conda/envs"
project = "$HOME/cin"

// Set script paths
script_convert  = "$project/src/czi2ometif.py"
script_cellpose = "$project/src/segmentation_cellpose.py"
script_stardist = "$project/src/segmentation_stardist.py"
script_deepcell = "$project/src/segmentation_deepcell.py"
script_predict  = "$project/src/model/prediction2.py"

// Set log message
log.info """\
     SEGMENTATION PIPELINE
     =========================
     input folder        = ${params.input}
     segmentation method = ${params.segmentation}
    """
    .stripIndent()

process CONVERT2TIF{
// 	errorStrategy 'ignore'
	conda "${conda}/aicsimageio"
	publishDir "${params.input}/ometif", mode: "copy"

	input:
	path (inputfile)

	output:
	path '*.ome.tif'

	script:
    def s = "${params.s}" ? "-s ${params.s}" : ""
    def c = "${params.c}" ? "-c ${params.c}" : ""

	"""
	python $script_convert -i $inputfile -o . $s $c

	"""
}

// Process for stardist segmentation
process SEGMENTATION_STARDIST{
    conda "${conda}/stardist"
    publishDir "${params.input}/segmentation/stardist", mode: "copy"

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
    conda "${conda}/cellpose"
    publishDir "${params.input}/segmentation/${params.segmentation}", mode: "copy"

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
    errorStrategy 'ignore'
    conda "${conda}/deepcell"
    publishDir "${params.input}/segmentation/${params.segmentation}", mode: "copy"

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

process PREDICTION{
    errorStrategy 'ignore'
	conda "${conda}/prediction"
	publishDir "${params.input}/predictions/${params.segmentation}_${params.rf}_${params.e}", mode: "move"

    // Define the inputs
    input:
    tuple path (mask), path (image)

    // Define outputs
    output:
    file "*.csv"

    script:
    def rf = params.rf ? "-rf  ${params.rf}": ""
    def size = params.size ? "-s ${params.size}" : ""
    def e = params.e ? "-e ${params.e}" : ""
	"""
	python $script_predict -i $image -m $mask -mod $params.model -d $params.device -o . $rf $size $e
	"""

}

workflow {
    // Getting input channel
    input_ch = Channel.fromPath("${params.input}/raw/*")

    // Convert to tif
    converted = CONVERT2TIF(input_ch)

    // Running segmentation process
    if (params.segmentation.contains("stardist")) {
        segmented = SEGMENTATION_STARDIST(converted)
        println "Using stardist segmentation"
    } else  if (params.segmentation.contains("cellpose")) {
        segmented = SEGMENTATION_CELLPOSE(converted)
        println "Using cellpose segmentation"
    } else if (params.segmentation.contains("deepcell")) {
        segmented = SEGMENTATION_DEEPCELL(converted)
        println "Using deepcell segmentation"
    } else {
        println "Please specify a segmentation method"
    }

    segmented = segmented.map{ m ->
    def base = m.baseName
    [mask:m,  image:file("${params.input}/ometif/${base}.ome.tif")]
    }
    // Predict
    PREDICTION(segmented)
}
