#!/usr/bin/env nextflow
nextflow.enable.dsl=2
params.input = ""
params.model = ""
params.device = "cpu"
<<<<<<< HEAD

project = "/Users/miguelibarra/PycharmProjects/cin/"
script = "$project/src/model/prediction.py"
=======
params.output = ""
script = "$HOME/cin/src/model/prediction.py"
>>>>>>> ad8e9a9... Addapation to the server

log.info """\
	 PREDICTION
	 =========================
	 input folder  = ${params.input}
	 model pathway = ${params.model}
	 device        = ${params.device}
	"""
	.stripIndent()

process PREDICTION{
    queue 1
	conda '/Users/miguelibarra/.miniconda3/envs/stable'
	publishDir "${params.input}/predictions", mode: "move"

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
    input_ch = Channel.fromPath("${params.input}/isonuc/*", type: "dir")
    PREDICTION(input_ch)
}

