#!/usr/bin/env nextflow
nextflow.enable.dsl=2
params.images = ""
params.masks = ""
params.output = ""
params.rf = "0.7"
params.e = "10"
params.s = "256 256"
script = "/Users/miguelibarra/PycharmProjects/cin/src/extract_single_nuclei.py"

log.info """\
	 NUCLEAR ISOLATION PIPELINE
	 =========================
	 input folder : ${params.images}
	 masks folder : ${params.masks}
	 output folder: ${params.output}
	 expansion    : ${params.e}
	 resize factor: ${params.rf}
	"""
	.stripIndent()

process NUCLEAR_ISOLATION{
 	// errorStrategy 'ignore'
	conda '/Users/miguelibarra/.miniconda3/envs/stable'

	publishDir "${params.output}/${mask.baseName}_e${params.e}_rf${params.rf}", mode: "move"

    // Define the inputs
    input:
    tuple path (mask), path (image)

    // Define outputs
    output:
    file "*.png"

    script:
	"""
	python $script -m $mask -i $image -s $params.s -e $params.e -rf $params.rf  -o ./$mask.baseName
	"""
}

workflow {
	mych = Channel.fromPath("${params.masks}")
    .map{ m ->
      def base = m.baseName
      [mask:m,  image:file("${params.images}/${base}.ome.tif")]
    }

  NUCLEAR_ISOLATION(mych)
}