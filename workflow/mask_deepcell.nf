#!/usr/bin/env nextflow
params.input = ""
params.deepcell = ""
params.mpp = "0.65"
params.compartment = ""

project = "/Users/miguelibarra/PycharmProjects/cin"
conda =  "/Users/miguelibarra/.miniconda3/envs/"

script_deepcell = "$project/src/segmentation_deepcell.py"

log.info """\
	 CELPOSE SEGMENTATION PIPELINE
	 =========================
	 input folder : ${params.input}
	"""
	.stripIndent()

process SEGMENTATION_DEEPCELL{
	errorStrategy 'ignore'
	conda '${conda}/deepcell'
	publishDir "${params.input}/segmentation/${params.deepcell}${params.compartment}", mode: "move"

	input:
	path (nuclear_image), stageAs: "input.ome.tif"

	output:
	path '*.tif'

	script:
	def deepcell = "${params.deepcell}" ? "-d ${params.deepcell}" : ""
	def compartment = "${params.compartment}" ? "-c ${params.compartment}" : ""

	"""
	python $script_deepcell -i input.ome.tif -o . -m $params.mpp $deepcell $compartment
	"""
}

workflow {
    input_ch = Channel.fromPath("${params.input}/ometif/*")
    SEGMENTATION_DEEPCELL(input_ch)
}