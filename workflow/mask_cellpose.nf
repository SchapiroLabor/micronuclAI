#!/usr/bin/env nextflow
params.input = ""
params.device = ""
params.cp_model = "nuclei"
params.cp_diameter = ""

conda = "$HOME/.conda/envs"
project = "$HOME/cin"
script = "$project/src/segmentation_cellpose.py"

log.info """\
	 CELLPOSE SEGMENTATION PIPELINE
	 =========================
	 input folder : ${params.input}
	"""
	.stripIndent()


process CELLPOSE_SEGMENTATION{
//	errorStrategy 'ignore'
	conda "${conda}/cellpose"
	publishDir "${params.input}/segmentation/cellpose_${params.cp_model}", mode: "move"
	
	input:
	path (nuclear_image), stageAs: "input.ome.tif"

	output:
	path '*.tif'

	script:
	def gpu = "${params.device}" ? "-g" : ""
	def cp_model = "${params.cp_model}" ? "-m ${params.cp_model}" : ""
	def cp_diameter = "${params.cp_diameter}" ? "-d ${params.cp_diameter}" : ""

	"""
	python $script -i input.ome.tif -o . $gpu $cp_model $cp_diameter
	"""
}


workflow {
    input_ch = Channel.fromPath("${params.input}/ometif/*")
	CELLPOSE_SEGMENTATION(input_ch)
}

