#source /home/prabhat.kc/anaconda3/base_env.sh
#source /home/prabhat.kc/anaconda3/hmri_env.sh 
#cd /projects01/didsr-aiml/prabhat.kc/code/GitRepo/mpi_sfrc/
# ----------------------------------------------------------
# executes for sfrc code development/selected sh
# On Tuning set with 5 images with ht as 0.33
#   on sharp test: 5+3+2+6+5 = 21
#	on smooth test: 16
# ----------------------------------------------------------
# 
# 
# bash +x demo_sfrc_run 'CT' '' 'sh' 1 
# bash +x demo_sfrc_run 'CT' '' 'sm' 47
mode=$1     # "CT" or "MRI"
data_opt=$2 # "sel" sfrc on tuning set or '' sfrc on test set
ker_opt=$3  # "sh" sharp kernel or "sm" smooth kernel for CT || "unet" or "plstv" for MRI
nranks=$4   # no. of processors for mpi
echo "modality is:" $mode
echo "testing data is:" $data_opt
echo "nprocs used is: " $nranks

if [[ "$mode" == "CT" ]]
then
  echo "kernel option is: " $ker_opt
  OUTPUT_FNAME="./results/CT/sm_srgan_${data_opt}_${ker_opt}_L067/"
  INPUT_FOLDER="./ctsr/results/test_${ker_opt}_L067/ua_ll_smSRGAN${data_opt}_in_x4/checkpoint-generator-20/"
  INPUT_GEN="test_${ker_opt}_L067_cnn"
  TARGET_GEN="test_${ker_opt}_L067_gt"
  time mpirun --mca btl ^openib -np $nranks \
  python main.py --input-folder ${INPUT_FOLDER} --output-folder ${OUTPUT_FNAME} --patch-size 'p64'   \
  --input-gen-folder ${INPUT_GEN} --target-gen-folder ${TARGET_GEN} \
  --img-format 'raw' --frc-threshold '0.5' --in-dtype 'uint16' \
  --anaRing --inscribed-rings --rNx 512 --apply-hann --mtf-space --ht 0.33 \
  --windowing 'soft' --save-patched-subplots
elif [[ "$mode" == "MRI" ]]
then
  INPUT_FOLDER="./mrsub/recon_data/"
  INPUT_GEN="masked_${data_opt}_${ker_opt}_recons_ood"
  TARGET_GEN="masked_${data_opt}_gt_ood"
  OUTPUT_FNAME="./results/MRI/ood_${data_opt}_${ker_opt}/"
  if [[ "$ker_opt" == "unet" ]]
  then
    HT=0.16
    echo "method is unet and ht is:" $HT
  elif [[ "$ker_opt" == "plstv" ]]
  then
    HT=0.17
    echo "method is plstv and ht is:" $HT
  else
    echo "Re-check method type cmd input for MRI. It can be unet or plstv"
    break
  fi
  time mpirun --mca btl ^openib -np $nranks \
  python main.py --input-folder ${INPUT_FOLDER} --output-folder ${OUTPUT_FNAME} --patch-size 'p48'   \
  --input-gen-folder ${INPUT_GEN} --target-gen-folder ${TARGET_GEN} \
  --img-format 'png' --frc-threshold '0.75' --in-dtype 'uint8'\
  --anaRing --inscribed-rings --apply-hann --ht $HT --windowing 'gray' --save-patched-subplots 
else
  echo ""
  echo "re-check cmd line options in file demo_sfrc_run.sh"
  echo ""
fi
