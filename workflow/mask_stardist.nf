#!/usr/bin/env nextflow
params.input = ""

project = "$HOME/cin"
conda =  "$HOME/.conda/envs/"

script_stardist = "$project/src/segmentation_stardist.py"

log.info """\
	 STARDIST SEGMENTATION PIPELINE
	 ==============================
	 input folder : ${params.input}
	"""
	.stripIndent()

process SEGMENTATION_STARDIST{
// 	errorStrategy 'ignore'
	conda "${conda}/stardist"
	publishDir "${params.input}/segmentation/stardist", mode: "move"

	input:
	path (nuclear_image)

	output:
	path '*.tif'

	script:
    def output = nuclear_image.baseName + "_mask.tif"

	"""
	python $script_stardist -i $nuclear_image -o $output
	"""
}

workflow {
    input_ch = Channel.fromPath("${params.input}/ometif/*")
    SEGMENTATION_STARDIST(input_ch)
}
