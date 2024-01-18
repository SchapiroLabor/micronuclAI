#!/usr/bin/env nextflow
nextflow.enable.dsl=2
params.input = ""
params.model = ""
params.device = "cpu"
params.segmentation = ""
params.size = "256 256"
params.rf = "0.6"
params.e = "25"

conda = "$HOME/.conda/envs"
project = "$HOME/cin"
script = "$project/src/model/prediction2.py"

log.info """\
	 PREDICTION
	 =========================
	 input folder  = ${params.input}
	 model pathway = ${params.model}
	 device        = ${params.device}
	 segmentation  = ${params.segmentation}
	"""
	.stripIndent()

process PREDICTION{
    // errorStrategy 'ignore'
	conda "${conda}/prediction"
	publishDir "${params.input}/predictions/${params.segmentation}", mode: "move"

    // Define the inputs
    input:
    tuple path (mask), path (image)

    // Define outputs
    output:
    file "*.csv"

    script:
    def rf = params.rf ? "-rf  ${params.rf}": ""
    def size = params.size ? "-s ${params.size}" : ""
    def e = params.e ? "-e ${params.e}" : ""
	"""
	python $script -i $image -m $mask -mod $params.model -d $params.device -o . $rf $size $e
	"""

}

workflow {
    input_ch = Channel.fromPath("${params.input}/segmentation/${params.segmentation}/*.tif")
    .map{ m ->
      def base = m.baseName
      [mask:m,  image:file("${params.input}/ometif/${base}.ome.tif")]
    }
    PREDICTION(input_ch)
}

