#!/usr/bin/env nextflow
params.input = ""
params.device = ""

script = "/Users/miguelibarra/PycharmProjects/cin/src/segmentation_cellpose.py"

log.info """\
	 CELPOSE SEGMENTATION PIPELINE
	 =========================
	 input folder : ${params.input}
	"""
	.stripIndent()


process CELLPOSE_SEGMENTATION{
	errorStrategy 'ignore'
	conda '/Users/miguelibarra/.miniconda3/envs/stable'
	publishDir "${params.input}/segmentation/cellpose", mode: "move"
	
	input:
	path (nuclear_image), stageAs: "input.ome.tif"

	output:
	path '*.tif'

	script:
	def gpu = "${params.device}" ? "-g" : ""

	"""
	python $script -i input.ome.tif -o . $gpu
	"""
}


workflow {
    input_ch = Channel.fromPath("${params.input}/ometif/*")
	CELLPOSE_SEGMENTATION(input_ch)
}

