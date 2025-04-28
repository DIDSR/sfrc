# updates in luna
# source /home/prabhat.kc/anaconda3/base_env.sh
# source /home/prabhat.kc/anaconda3/hmri_env.sh 
# cd /projects01/didsr-aiml/prabhat.kc/code/mpi_sfrc/v2/
artifact=$1
nranks=$2
INFOLDER='conventional_artifact'
if [[ "$artifact" == "missing_wedge" ]]
then
  echo "artifact option is: " $artifact
  echo "nprocs used is: " $nranks
  OUTPUT_FNAME="./results/${artifact}/"
  INPUT_FOLDER="./${INFOLDER}/data/${artifact}/"
  INPUT_GEN="input_irt"
  TARGET_GEN="gt_irt"
  mpirun --mca btl ^openib -np $nranks \
  python main.py --input-folder ${INPUT_FOLDER} --output-folder ${OUTPUT_FNAME} --patch-size 'p64'   \
  --input-gen-folder ${INPUT_GEN} --target-gen-folder ${TARGET_GEN} \
  --img-format 'raw' --frc-threshold '0.5' --in-dtype 'uint16' \
  --anaRing --inscribed-rings --rNx 512 --apply-hann --mtf-space --ht 0.33 \
  --windowing 'L260_W780_soft_artifact' --save-patched-subplots --rNx 512

elif [[ "$artifact" == "distortion" ]]
then
  echo "artifact option is: " $artifact
  echo "nprocs used is: " $nranks
  OUTPUT_FNAME="./results/${artifact}/"
  INPUT_FOLDER="./${INFOLDER}/data/${artifact}/"
  INPUT_GEN="input_irt"
  TARGET_GEN="gt_irt"
  mpirun --mca btl ^openib -np $nranks \
  python main.py --input-folder ${INPUT_FOLDER} --output-folder ${OUTPUT_FNAME} --patch-size 'p64'   \
  --input-gen-folder ${INPUT_GEN} --target-gen-folder ${TARGET_GEN} \
  --img-format 'raw' --frc-threshold '0.5' --in-dtype 'uint16' \
  --anaRing --inscribed-rings --rNx 512 --apply-hann --mtf-space --ht 0.33 \
  --windowing 'L260_W780_soft_artifact' --save-patched-subplots --rNx 512

elif [[ "$artifact" == "blur" ]]
then
  echo "artifact option is: " $artifact
  echo "nprocs used is: " $nranks
  OUTPUT_FNAME="./results/${artifact}/bc/"
  INPUT_FOLDER="./${INFOLDER}/data/${artifact}/checkpoint-generator-20/"
  INPUT_GEN="blur_bc"
  TARGET_GEN="blur_gt"
  mpirun --mca btl ^openib -np $nranks \
  python main.py --input-folder ${INPUT_FOLDER} --output-folder ${OUTPUT_FNAME} --patch-size 'p64'   \
  --input-gen-folder ${INPUT_GEN} --target-gen-folder ${TARGET_GEN} \
  --img-format 'raw' --frc-threshold '0.5' --in-dtype 'uint16' \
  --anaRing --inscribed-rings --rNx 512 --apply-hann --mtf-space --ht 0.33 \
  --windowing 'L260_W780_soft_artifact' --save-patched-subplots --rNx 512

elif [[ "$artifact" == "noise" ]]
then
  echo "artifact option is: " $artifact
  echo "nprocs used is: " $nranks
  OUTPUT_FNAME="./results/${artifact}/"
  INPUT_FOLDER="./${INFOLDER}/data/${artifact}/"
  INPUT_GEN="input_irt"
  TARGET_GEN="gt_irt"
  mpirun --mca btl ^openib -np $nranks \
  python main.py --input-folder ${INPUT_FOLDER} --output-folder ${OUTPUT_FNAME} --patch-size 'p64'   \
  --input-gen-folder ${INPUT_GEN} --target-gen-folder ${TARGET_GEN} \
  --img-format 'raw' --frc-threshold '0.5' --in-dtype 'uint16' \
  --anaRing --inscribed-rings --rNx 512 --apply-hann --mtf-space --ht 0.33 \
  --windowing 'L260_W780_soft_artifact' --save-patched-subplots --rNx 512

else
  echo ""
  echo "re-check cmd line options in file non_fake_artifacts.sh"
  echo ""
fi
