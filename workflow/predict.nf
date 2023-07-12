#!/usr/bin/env nextflow
nextflow.enable.dsl=2
params.input = ""
params.model = ""
params.device = "cpu"
params.segmentation = "cellpose_nuclei"

conda = "$HOME/.conda/envs"
project = "$HOME/cin"
script = "$project/src/model/prediction.py"

log.info """\
	 PREDICTION
	 =========================
	 input folder  = ${params.input}
	 model pathway = ${params.model}
	 device        = ${params.device}
	"""
	.stripIndent()

process PREDICTION{
	conda "${conda}/pytorch_lt"
	publishDir "${params.input}/predictions/${params.segmentation}", mode: "move"

    input:
    path (images)

    // Define outputs
    output:
    file "*.csv"

    script:
	"""
	python $script -i $images -o . -d $params.device -m $params.model
	"""

}

workflow {
    input_ch = Channel.fromPath("${params.input}/isonuc/${params.segmentation}/*", type: "dir")
    PREDICTION(input_ch)
}

