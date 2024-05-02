#!/usr/bin/env nextflow
params.input = ""
params.segmentation = ""
params.mpp = ""
params.batch_size = ""

project = "$HOME/cin"
conda =  "$HOME/.conda/envs/"

script_deepcell = "$project/src/segmentation_deepcell.py"

log.info """\
	 DEEPCELL SEGMENTATION PIPELINE
	 =========================
	 input folder : ${params.input}
	"""
	.stripIndent()

process SEGMENTATION_DEEPCELL{
// 	errorStrategy 'ignore'
	conda "${conda}/deepcell"
	publishDir "${params.input}/segmentation/deepcell_${params.segmentation}", mode: "move"

	input:
	path (nuclear_image)

	output:
	path '*.tif'

	script:
	def output = nuclear_image.baseName + "_mask.tif"
	def model = "${params.segmentation}" ? "-m ${params.segmentation}" : ""
	def batch_size = "${params.batch_size}" ? "-bs ${params.batch_size}" : ""
	def mpp = "${params.mpp}" ? "-mpp ${params.mpp}" : ""

	"""
	python $script_deepcell -i input.ome.tif -o $output -mpp $params.mpp $model $batch_size
	"""
}

workflow {
    input_ch = Channel.fromPath("${params.input}/ometif/*")
    SEGMENTATION_DEEPCELL(input_ch)
}
