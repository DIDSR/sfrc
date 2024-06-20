
#------------------------------------------------------------------------------------------------------------------#
#                                             OPTIONS
#------------------------------------------------------------------------------------------------------------------#
<< COMMENT
usage: resolve_sr.py [-h] [--model-name MODEL_NAME] --input-folder INPUT_FOLDER [--gt-folder GT_FOLDER] --model-folder MODEL_FOLDER
                  [--output-folder OUTPUT_FOLDER] --normalization-type NORMALIZATION_TYPE [--cuda] [--input-img-type INPUT_IMG_TYPE] 
                  [--specific-epoch] [--chckpt-no CHCKPT_NO] [--se-plot] [--in-dtype IN_DTYPE] [--out-dtype OUT_DTYPE] [--resolve-nps]

PyTorch application of trained weight on CT images

arguments:
  -h, --help            show this help message and exit
  --model-name , --m    choose the network architecture name that you are
                        going to use. Other options include redcnn, dncnn,
                        unet, gan.
  --input-folder        directory name containing noisy input test images.
  --gt-folder           directory name containing test Ground Truth images.
  --model-folder        directory name containing saved checkpoints.
  --output-folder       path to save the output results.
  --normalization-type  None or unity_independent. Look into
                        img_pair_normalization in utils.
  --cuda                use cuda
  --input-img-type      dicom or raw or tif?
  --specific-epoch      If true only one specific epoch based on the chckpt-no
                        will be applied to test images. Else all checkpoints
                        (or every saved checkpoints corresponding to each
                        epoch) will be applied to test images.
  --chckpt-no           epoch no. of the checkpoint to be loaded and then
                        applied to noisy images from the test set. Default is
                        the last epoch.
  --se-plot             If true denoised images from test set is saved inside
                        the output-folder. Else only test stats are saved in
                        .txt format inside the output-folder.
  --in-dtype            data type to save de-noised output.
  --out-dtype           data type to save de-noised output.
  --resolve-nps         is CNN applied to water phantom images?
  --resolve-patient     is CNN applied to images from different patients? If
                        yes then images will be saved with patient tag.
  --resolve-nps         is CNN applied to water phantom images?
  --rNx RNX             image size for raw image as input (specifically for
                        LR).
  --scale SCALE         up-scaling factor.
  

COMMENT
#source /home/prabhat.kc/anaconda3/base_env.sh
#source /home/prabhat.kc/anaconda3/horovod_env.sh
#cd /projects01/didsr-aiml/prabhat.kc/code/GitRepo/mpi_sfrc/ctsr

data_opt=$1 #data option {sel: tuning set or '':entire testset}}
ker_opt=$2 #kernel option {sh: sharp or sm: smooth}
# to get entire testset look into create_sr_testset/readme.txt

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

RESOLVE_PY=resolve_sr.py
MODEL_FOLDER="./checkpoints/srgan/"
NORM_TYPE="unity_independent"
INPUT_FOLDER="./data/test_${ker_opt}_L067/ua_ll_3mm_LR_${data_opt}_x4/"
GT_FOLDER="./data/test_${ker_opt}_L067/ua_ll_3mm_HR_${data_opt}_x4/"
OUTPUT_FOLDER="./results/test_${ker_opt}_L067/ua_ll_smSRGAN$_{data_opt}_in_x4/"

python $RESOLVE_PY --m 'srgan' --input-folder ${INPUT_FOLDER} --model-folder ${MODEL_FOLDER} --gt-folder ${GT_FOLDER} \
--output-folder ${OUTPUT_FOLDER} --cuda --normalization-type $NORM_TYPE --input-img-type 'raw' --specific-epoch \
--resolve-patient --rNx 128 --scale 4 --se-plot