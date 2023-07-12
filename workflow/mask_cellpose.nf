#!/usr/bin/env nextflow
params.input = ""
params.device = ""
parmas.cp_model = "nuclei"

conda = "$HOME/.conda/envs"
project = "$HOME/cin"
script = "$project/src/segmentation_cellpose.py"

log.info """\
	 CELPOSE SEGMENTATION PIPELINE
	 =========================
	 input folder : ${params.input}
	"""
	.stripIndent()


process CELLPOSE_SEGMENTATION{
	errorStrategy 'ignore'
	conda '${conda}/stable'
	publishDir "${params.input}/segmentation/cellpose_${params.cp_model}", mode: "move"
	
	input:
	path (nuclear_image), stageAs: "input.ome.tif"

	output:
	path '*.tif'

	script:
	def gpu = "${params.device}" ? "-g" : ""

	"""
	python $script -i input.ome.tif -o . $gpu $params.cp_model
	"""
}


workflow {
    input_ch = Channel.fromPath("${params.input}/ometif/*")
	CELLPOSE_SEGMENTATION(input_ch)
}

