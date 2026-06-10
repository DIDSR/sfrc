* For using sFRC

  - The image reconstruction packages described below are not required for running sFRC. Simply follow the installation instructions provided at https://github.com/DIDSR/sfrc/tree/master#requirements. After installation, the demo sFRC file (``demo_sfr_run.sh``) can be run using the reference and undersampled radiological images included in this repository.


* For applying GAN, WGAN, and U-Net pretrained weights

  - Using ``python==3.7.5``, we used ``torch==1.6.0``, ``torchvision==0.7.0``, and ``pytorch-lightning==0.7.3`` to apply the U-Net pretrained weights provided in the `hallucinations-tomo-recon repository <https://github.com/comp-imaging-sci/hallucinations-tomo-recon>`_ for the deep learning–based MR post-processing task.

  - Additional Python package versions used include ``certifi==2019.11.28``, ``cffi==1.13.2``, ``cloudpickle==1.3.0``, ``cycler==0.10.0``, ``dask==2.12.0``, ``decorator==4.4.2``, ``kiwisolver==1.2.0``, ``more-itertools==4.3.0``, ``networkx==2.4``, ``numpy==1.18.2``, ``olefile==0.46``, ``Pillow==6.2.1``, ``pluggy==0.8.0``, ``protobuf==3.6.1``, ``py==1.7.0``, ``pycparser==2.19``, ``pyparsing==2.4.6``, ``python-dateutil==2.8.1``, ``pytz==2019.3``, ``runstats==1.8.0``, ``six==1.13.0``, ``tensorboard==1.14.0``, ``toolz==0.10.0``, ``tornado==6.1.0``, ``test-tube==0.7.3``, ``natsort==8.4.0``, and ``opencv-python==4.8.1.78``.

  - The same Python configuration is compatible with the WGAN and GAN pretrained weights provided in this repository for the CT super-resolution task.


* For GAN and WGAN training

  - Training the WGAN and GAN models requires Horovod. For instructions on Horovod-based multi-GPU training, refer to the `Horovod documentation <https://horovod.readthedocs.io/en/latest/install_include.html>`_.


* For CT PAIL testing and training

  - Refer to the `PAIL repository <https://github.com/seuzjj/PAIL>`_.