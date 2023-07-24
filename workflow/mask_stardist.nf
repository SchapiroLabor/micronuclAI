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
	path (nuclear_image), stageAs: "input.ome.tif"

	output:
	path '*.tif'

	script:
// 	def model = "${params.segmentation}" ? "-m ${params.segmentation}" : ""
// 	def batch_size = "${params.batch_size}" ? "-bs ${params.batch_size}" : ""
// 	def mpp = "${params.mpp}" ? "-mpp ${params.mpp}" : ""

	"""
	python $script_stardist -i input.ome.tif -o .
	"""
}

workflow {
    input_ch = Channel.fromPath("${params.input}/ometif/*")
    SEGMENTATION_STARDIST(input_ch)
}
