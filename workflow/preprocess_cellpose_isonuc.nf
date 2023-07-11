#!/usr/bin/env nextflow
params.input = ""
params.c = ""
params.s = ""

params.gpu = ""

params.rf = "0.7"
params.e = "10"
params.size = "256 256"

project = "$HOME/PycharmProjects/cin"
conda =  "$HOME/.miniconda3/envs/"

params.model = "$project/models/binary_10K/models/model_1.pt"
params.device = "mps"

// Scripts
script_convert = "$project/src/czi2ometif.py"
script_cellpose = "$project/src/segmentation_cellpose.py"
script_isonuc = "$project/src/extract_single_nuclei.py"
script_prediction = "$project/src/model/prediction.py"



log.info """\
	 MICRONUCL(AI) PIPELINE
	 =========================
	 input folder : ${params.input}
	"""
	.stripIndent()


process CONVERT2OMETIF{
	errorStrategy 'ignore'
<<<<<<< HEAD
	conda '${conda}/aicsimageio'
=======
	conda '/home/hd/hd_hd/hd_hm294/.conda/envs/aicsimageio'
>>>>>>> 77112d2... for usage on cluster
	publishDir "${params.input}/ometif", mode: "copy"

	input:
	path (input_file)

	output:
	path '*.ome.tif'

	script:
    def s = "${params.s}" ? "-s ${params.s}" : ""
    def c = "${params.c}" ? "-c ${params.c}" : ""

	"""
	python $script_convert -i $input_file -o . $s $c

	"""
}

process CELLPOSE_SEGMENTATION{
// 	errorStrategy 'ignore'
<<<<<<< HEAD
	conda '${conda}/stable'
=======
	queue 1
	conda '/home/hd/hd_hd/hd_hm294/.conda/envs/cellpose'
>>>>>>> 77112d2... for usage on cluster
	publishDir "${params.input}/segmentation/cellpose", mode: "copy"

	input:
	path (nuclear_image), stageAs: "input.ome.tif"

	output:
	path '*.tif'

	script:
	def gpu = "${params.gpu}" ? "-g" : ""

	"""
	python $script_cellpose -i input.ome.tif -o . $gpu
	"""
}

process NUCLEAR_ISOLATION{
//  	errorStrategy 'ignore'
<<<<<<< HEAD
	conda '${conda}/stable'
=======
	conda '/home/hd/hd_hd/hd_hm294/.conda/envs/cellpose'
>>>>>>> 77112d2... for usage on cluster
	publishDir "${params.input}/isonuc/", mode: "copy"

    // Define the inputs
    input:
    path (mask)

    // Define outputs
    output:
    path "${mask.baseName}_e${params.e}_rf${params.rf}"

    script:
	"""
	mkdir "./${mask.baseName}_e${params.e}_rf${params.rf}"
	python $script_isonuc -m "${mask}" -i "${params.input}/ometif/${mask.baseName}.ome.tif" -s $params.size -e $params.e -rf $params.rf  -o "./${mask.baseName}_e${params.e}_rf${params.rf}/${mask.baseName}"
	"""
}

process PREDICTION{
	queue 1
 	errorStrategy 'ignore'
<<<<<<< HEAD
	conda '${conda}/stable'
=======
	conda '/home/hd/hd_hd/hd_hm294/.conda/envs/pytorch_lt'
>>>>>>> 77112d2... for usage on cluster
	publishDir "${params.input}/predictions/", mode: "move"

    input:
    path (images)

    // Define outputs
    output:
    file "*.csv"

    script:
	"""
	python $script_prediction -i $images -o . -d $params.device -m $params.model
	"""

}


workflow {
    // Input channel
    input_ch = Channel.fromPath("${params.input}/raw/*")

    // Convert to ome.tif
	nuclear_image = CONVERT2OMETIF(input_ch)

	// Cellpose segmentation
	masks = CELLPOSE_SEGMENTATION(nuclear_image)

	// NUCLEAR_ISOLATION
    isonucs = NUCLEAR_ISOLATION(masks)

    // PREDICTION
    PREDICTION(isonucs)
}

