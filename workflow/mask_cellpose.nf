#!/usr/bin/env nextflow
params.input = ""
params.device = "cpu"
params.segmentation = ""
params.cp_diameter = ""
params.batch_size = ""

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
	publishDir "${params.input}/segmentation/cellpose_${params.segmentation}", mode: "move"
	
	input:
	path (nuclear_image)

	output:
	path '*_mask.tif'

	script:
	def output = nuclear_image.baseName + "_mask.tif"
	def segmentation = "${params.segmentation}" ? "-m ${params.segmentation}" : ""
	def batch_size = "${params.batch_size}" ? "-bs ${params.batch_size}" : ""
	def cp_diameter = "${params.cp_diameter}" ? "-dm ${params.cp_diameter}" : ""

	"""
	python $script -i input.ome.tif -o $output -d $params.device $segmentation $cp_diameter $batch_size
	"""
}


workflow {
    input_ch = Channel.fromPath("${params.input}/ometif/*")
	CELLPOSE_SEGMENTATION(input_ch)
}

