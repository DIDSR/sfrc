condaenv=$1

if [[ "$condaenv" == "mpi_sfrc" ]]
then
  echo "creating conda enviroment and installing required packages for sfrc-based computations"
  conda create -n mpi_sfrc python=3.7.5 --no-default-packages
  conda activate mpi_sfrc
  #chmod +x /home/prabhat.kc/anaconda3/envs/mpi_sfrc/bin/pip
  #chmod +x /home/prabhat.kc/anaconda3/envs/mpi_sfrc/bin/*
  conda install -c anaconda h5py==3.6.0
  pip install PyWavelets==1.3.0
  pip install -r ./mrsub/unet/requirements.txt
  pip install pydicom==2.4.3
  pip install opencv-python==4.8.1.78
  pip install jupyterlab==3.6.6
  pip install mpi4py==3.1.5
  pip install bm3d==4.0.1
  pip install shapely==2.0.2
elif [[ "$condaenv" == "srgan" ]]
then
  echo "creating conda enviroment and installing required packages for srgan-based computations"
elif [[ "$condaenv" == "vhmri" ]]
then
  echo "creating conda enviroment and installing required packages for unet-based computations"
else
  echo ""
  echo "re-check cmd line options in file demo_sfrc_run.sh"
  echo ""
fi