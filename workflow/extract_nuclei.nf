#!/usr/bin/env nextflow
nextflow.enable.dsl=2
params.input = ""
params.segmentation = ""
params.rf = "0.6"
params.e = "10"
params.s = "256 256"
params.fb = ""

conda = "$HOME/.conda/envs"
project = "$HOME/cin"
script = "$project/src/extract_single_nuclei.py"

log.info """\
	 NUCLEAR ISOLATION PIPELINE
	 =========================
	 input folder : ${params.input}
	 segmentation : ${params.segmentation}
	 resize factor: ${params.rf}
	 expansion    : ${params.e}
	 size         : ${params.s}
	"""
	.stripIndent()

process NUCLEAR_ISOLATION{
 	// errorStrategy 'ignore'
	conda "${conda}/mask2bbox"
	publishDir "${params.input}/isonuc/${params.segmentation}/${mask.baseName}_e${params.e}_rf${params.rf}", mode: "move"

    // Define the inputs
    input:
    tuple path (mask), path (image)

    // Define outputs
    output:
    file "*.png"

    script:
    def fb = params.fb ? "-fb" : ""
	"""
	python $script -m $mask -i $image -s $params.s -e $params.e -rf $params.rf  -o ./$mask.baseName $fb
	"""
}

workflow {
	mych = Channel.fromPath("${params.input}/segmentation/${params.segmentation}/*.tif")
    .map{ m ->
      def base = m.baseName
      [mask:m,  image:file("${params.input}/ometif/${base}.ome.tif")]
    }

  NUCLEAR_ISOLATION(mych)
}
