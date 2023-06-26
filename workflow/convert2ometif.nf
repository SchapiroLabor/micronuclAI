#!/usr/bin/env nextflow
params.input = ""
params.output = ""
params.c = ""
params.s = ""

script = "/Users/miguelibarra/PycharmProjects/cin/src/czi2ometif.py"
input_ch = Channel.fromPath(params.input)

log.info """\
	 CZI 2 OME.TIF-NF PIPELINE
	 =========================
	 input folder : ${params.input}
	 output folder: ${params.output}
	 channel      : ${params.channel}
	"""
	.stripIndent()


process CONVERTCZI2TIF{
// 	errorStrategy 'ignore'
// 	conda '/Users/miguelibarra/.miniconda3/envs/aicsimageio'
    conda '/Users/miguelibarra/PycharmProjects/cin/micronucleai.yml'
	publishDir "${params.output}", mode: "move"
	
	input:
	path (czi_file)

	output:
	path '*.ome.tif'

	script:
    def s = "${params.s}" ? "-s ${params.s}" : ""
    def c = "${params.c}" ? "-c ${params.c}" : ""

	"""
	python $script -i $czi_file -o . $s $c
	
	"""
}


workflow {
	CONVERTCZI2TIF(input_ch)
}

