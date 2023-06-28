 #!/usr/bin/env nextflow
nextflow.enable.dsl=2
params.images = ""
params.model = ""
params.device = "cpu"
params.output = ""
script = "/Users/miguelibarra/PycharmProjects/cin/src/model/prediction.py"

log.info """\
	 PREDICTION
	 =========================
	 input folder : ${params.images}
	 model pathway : ${params.model}
	 output folder: ${params.output}
	"""
	.stripIndent()

process PREDICTION{
 	errorStrategy 'ignore'
	conda '/Users/miguelibarra/.miniconda3/envs/stable'
	publishDir "${params.output}", mode: "move"

    // Define outputs
    output:
    file "*.csv"

    script:
	"""
	python $script -i $params.images -o . -d $params.device -m $params.model
	"""

}

workflow {
    PREDICTION()
}

