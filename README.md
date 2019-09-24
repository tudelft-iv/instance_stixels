# Description
This repository contains an implementation of the Instance Stixel pipeline
which was introduced in [our](http://www.intelligent-vehicles.org)
corresponding paper:
[Instance Stixels: Segmenting and Grouping Stixels into Objects](http://intelligent-vehicles.org/wp-content/uploads/2019/05/hehn2019iv_instance_stixels.pdf)

If you use our work, please cite:
```
@inproceedings{Hehn2019,
    title     = {Instance Stixels: Segmenting and Grouping Stixels into Objects},
    author    = {Thomas Hehn and Julian F.P. Kooij and Dariu M. Gavrila},
    booktitle = {IEEE Intelligent Vehicles Symposium (IV), Paris (France)},
    year      = {2019},
}
```

Disclaimer:
This implementation is a prototype and consists of multiple separate programs.
It is not optimized for efficiency, but to provide a proof of concept and
initial starting point for further research and development in this area.
This program is distributed in the hope that it will be useful, but without any
warranty.

# Requirements

### C++

* CUDA (Note hardware requirements below)
* HDF5
* OpenCV
* cmake

Except for cuda, this should do the trick on Ubuntu:
```
apt-get install -y cmake libopencv-dev libhdf5-dev
```


### Python/Conda

At the moment conda is required.
It is recommended to install miniconda for python 3.x from:
<https://docs.conda.io/en/latest/miniconda.html>.
You can then create the conda environment from the yml file as follows:
```
conda env create -f instance_stixel_env.yml
```

Note: Running the pipeline without conda will require some modifications of the
bash scripts.

### Other (just for completeness)

* make, bash, awk, ... Linux users should be fine.

### Hardware (GPU)

The pipeline was tested on a NVIDIA Titan V.
The code requires about 56KB of shared memory for 1792x784 Cityscapes images.
Downscaling the images or cropping may help to make it work on other cards
with less shared memory available. However, this has not been tested.

# Installation

### Compile GPUStixels

Assuming you have installed all the dependencies listed in the requirements
section (including the conda environment "instance_stixels"), you only need to
compile the GPUStixels code.
Do this as follows:
```
cd GPUStixels
mkdir build
cd build
cmake ..
make
```
Afterwards, you can run a short test to be sure:
```
make test
```

*Note*: If you only get a failed test, you might need to adapt the `CUDA_NVCC_FLAGS` in the `CMakeLists.txt` to build for your GPUs compute capabilities.

### Download CNN weights

Download the CNN weights and save them to `instanceoffset/weights/`:
* `drn_d_22_cityscapes.pth` from http://go.yf.io/drn-cityscapes-models
* `Net_DRNRegDs_epoch45.pth` from https://surfdrive.surf.nl/files/index.php/s/7r8QEbTb1hcyvOS

# Testing & running
### Test on cityscapes

To run the provided test script, you need the following files of Cityscapes:
* leftImg8bit_trainvaltest.zip (11GB)
* disparity_trainvaltest.zip (3.5GB)
* gtFine_trainvaltest.zip (241MB)
* camera_trainvaltest.zip (2MB)

You can download these at https://www.cityscapes-dataset.com/downloads/.
You have to register there to get access.
Extract them in a single folder (the cityscapes root folder), 
which should automatically contain the subfolders:
* leftImg8bit/train
* disparity/train
* camera/train
* gtFine/train

Now, set the variable `CITYSCAPES_PATH` in `run.py` to point to the cityscapes
root folder.
You can then test the whole pipeline as follows:
```
bash tests/run_test.sh verbose
```

### Running
See the bash script `tests/run_test.sh` as an example of how to use `run.py`
and checkout its command line help `python3 run.py --help`.

# Acknowlegdement

Instance Stixels builds upon a variety of open source software. We would like
to thank the authors and contributors of the following projects for sharing
their code!
* GPU Stixels: https://github.com/dhernandez0/stixels
* Dilated Residual Network: https://github.com/fyu/drn
* Catch2: https://github.com/catchorg/Catch2
* RapidJSON: http://rapidjson.org/

# Contact 

See
[paper](http://intelligent-vehicles.org/wp-content/uploads/2019/05/hehn2019iv_instance_stixels.pdf)
for the email address of the corresponding author or go to
http://www.intelligent-vehicles.org.
