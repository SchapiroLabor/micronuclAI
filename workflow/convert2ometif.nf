#!/usr/bin/env nextflow
params.input = ""
params.c = ""
params.s = ""

conda = "$HOME/.conda/envs"
project = "$HOME/cin"
script = "$project/src/czi2ometif.py"

log.info """\
	 CONVERT 2 OME.TIF-NF PIPELINE
	 =========================
	 input folder : ${params.input}
	"""
	.stripIndent()


process CONVERTCZI2TIF{
// 	errorStrategy 'ignore'
	conda '${conda}/aicsimageio'
	publishDir "${params.input}/ometif", mode: "move"
	
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
    input_ch = Channel.fromPath("${params.input}/raw/*")
	CONVERTCZI2TIF(input_ch)
}

