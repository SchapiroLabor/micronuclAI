#!/usr/bin/env nextflow
params.input = ""
params.output = ""
params.gpu = ""

script = "/Users/miguelibarra/PycharmProjects/cin/src/segmentation_cellpose.py"

log.info """\
	 CELPOSE SEGMENTATION PIPELINE
	 =========================
	 input folder : ${params.input}
	 output folder: ${params.output}
	"""
	.stripIndent()


process CELLPOSE_SEGMENTATION{
	errorStrategy 'ignore'
	conda '/Users/miguelibarra/.miniconda3/envs/stable'
	publishDir "${params.output}", mode: "move"
	
	input:
	path (nuclear_image), stageAs: "input.ome.tif"

	output:
	path '*.tif'

	script:
	def gpu = "${params.gpu}" ? "-g" : ""

	"""
	python $script -i input.ome.tif -o . $gpu
	"""
}


workflow {
    input_ch = Channel.fromPath(params.input)
	CELLPOSE_SEGMENTATION(input_ch)
}

