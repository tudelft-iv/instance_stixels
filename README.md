# Description
This repository contains an implementation of the Instance Stixel pipeline
which was presented in [our](http://www.intelligent-vehicles.org)
corresponding papers
[Fast and Compact Image Segmentation using Instance Stixels (T-IV, 2021)](http://intelligent-vehicles.org/wp-content/uploads/2021/03/hehn2021tiv_instance_stixels.pdf)
and
[Instance Stixels: Segmenting and Grouping Stixels into Objects (IV, 2019)](http://intelligent-vehicles.org/wp-content/uploads/2019/05/hehn2019iv_instance_stixels.pdf).
Our [video](https://www.youtube.com/watch?v=irrPsoWQoLY) demonstrates the
segmentation results obtained using Instance Stixels.

If you use our work, please cite the Journal paper:
```
@ARTICLE{hehn2021,
  author={T. {Hehn} and J. F. P. {Kooij} and D. M. {Gavrila}},
  journal={IEEE Transactions on Intelligent Vehicles},
  title={Fast and Compact Image Segmentation using Instance Stixels},
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIV.2021.3067223}
}
```

# Overview

The easiest way to use Instance Stixels is by using our
[singularity image](https://surfdrive.surf.nl/files/index.php/s/UMv4wyf200R7Kio)
(~7.2 GB).
The only requirements for the host system are
[singularity](https://sylabs.io/guides/3.6/user-guide/quick_start.html#quick-installation-steps)
(>=3.6.1, some older 3.x
versions may also work) and a NVIDIA cuda driver (>=10.2, see CUDA Version in
`nvidia-smi` output).
Further, we tested Instance Stixels only with the following GPUs:
NVIDIA Titan Xp, Titan V, and Titan RTX.

We provide two independent applications to use our Instance Stixels library:
1. A run script to evaluate Instance Stixels on the Cityscapes dataset
   ([tools/run_run_cityscapes.py](tools/run_run_cityscapes.py))
2. A ROS node for online processing ([apps/stixels_node_main.cu](apps/stixels_node_main.cu))

Both applications are available via the singularity image.

## Pre-trained CNN weights
For both applications you will need to download the
[CNN weights files](https://surfdrive.surf.nl/files/index.php/s/7IaK38xq1SZlSec)
first! In the following `<WEIGHTS_PATH>` will refer to the directory where the
zip file was extracted. It should contain the following files:
```
<WEIGHTS_PATH>/
    drn_d_38/DRNDSDoubleSegSL_0.0001_0.0001_0_0_0095.pth
    drn_d_22/DRNDSDoubleSegSL_1e-05_0.0001_0_0_0065.pth
    onnx/DRNDSDoubleSegSL_0.0001_0.0001_0_0_0095_zmuv_fp.onnx
    onnx/DRNDSDoubleSegSL_1e-05_0.0001_0_0_0065_zmuv_fp.onnx
```

## Evaluation on Cityscapes

First make sure to download the [Cityscapes dataset](https://www.cityscapes-dataset.com/),
especially the files:
- camera_trainvaltest.zip (2MB)
- leftImg8bit_trainvaltest.zip (11GB)
- disparity_trainvaltest.zip (3.5GB)
- gtFine_trainvaltest.zip (241MB)

The Instance Stixels run script for Cityscapes expects the following folder
structure:
```
<CITYSCAPES_PATH>/
    camera/train/
        aachen/
            aachen_*_*_camera.json
        ...
    camera/test/
        aachen/
            aachen_*_*_camera.json
        ...
    leftImg8bit/train/
        aachen/
            aachen_*_*_leftImg8bit.png
        ...
    leftImg8bit/test/
        aachen/
            aachen_*_*_leftImg8bit.png
        ...
    disparity/train/
        aachen/
            aachen_*_*_leftImg8bit.png
        ...
    disparity/test/
        aachen/
            aachen_*_*_leftImg8bit.png
        ...
    gtFine/train/
        aachen/
            aachen_*_*_gtFine_labelsIds.png
            aachen_*_*_gtFine_instanceIds.png
        ...
    gtFine/test/
        aachen/
            aachen_*_*_gtFine_labelsIds.png
            aachen_*_*_gtFine_instanceIds.png
        ...
```

You can now run the script [tests/run_test.sh](tests/run_test.sh) within the
singularity container and bind the `<CITYSCAPES_PATH>` and `<WEIGHTS_PATH>`
from the host as follows:
```
singularity run --app cityscapes_test --nv -B <CITYSCAPES_PATH>:/data/Cityscapes -B <WEIGHTS_PATH>:/data/weights instance-stixels.sif [long] [unary] [verbose]
```

The optional argument `long` will run the test script on the
validation set of the cityscapes dataset. Thus, it will reproduce the results
of Table I in our Journal publication.
After running the test you can find visualizations of the
results in the directory `~/.tmp/instance-stixels/long_test/stixelsim/`.
The short test works simply with a subset of the validation data and is much
faster.

Furthermore, you can shell into the singularity container to run your own tests in
the singularity container without installing any additional dependencies.
You can use the script [tools/run_cityscapes.py](tools/run_cityscapes.py) for
this.

*Notes*:
- This command will write temporary files to a `${HOME}/.tmp` directory on
the host system.
- The tests may fail when there are slight differences in the segmentation
result (this usually only affects the average number of stixels).
We experienced slight differences between different GPUs.
Our results were obtained using a Titan V GPU.

## ROS Node

We provide a ROS node that segments images on-the-fly using Instance Stixels
and outputs instance stixel messages (see below) as well as 3D marker arrays
and images as visualizations. You can find a short video that highlights some
of the features of the ROS node
[here](https://surfdrive.surf.nl/files/index.php/s/HcSrUeUzVSnadwI)
(this was an older version and more features have been added since then).

The singularity image provides a simple way to run the
[launch file](launch/instance_stixels.launch)
of the ROS node in the singularity container:
```
singularity run --app ros_node --nv instance-stixels.sif onnxfilename:=/<WEIGHTS_PATH>/onnx/DRNDSDoubleSegSL_1e-05_0.0001_0_0_0065_zmuv_fp.onnx camera_id:=/<camera_id>
```

Running the nodes in the singularity container still enables you to communicate
with the ROS node from ROS running on the host.
This also means that you can use the ROS node in the container to process a
rosbag played from the host system!

When you use the command above, the ROS node subscribes to the following topics:
- `/<camera_id>/disparity` [stereo_msgs/DisparityImage]
- `/<camera_id>/left/image_color` [sensor_msgs/Image]
- `/<camera_id>/left/camera_info` [sensor_msgs/CameraInfo
- `/<camera_id>/right/camera_info` [sensor_msgs/CameraInfo]

From the incoming left image and disparity image, the node will extract a
1792x784 region of interest
(see code in [apps/stixels_node.cu#L161](./apps/stixels_node.cu#L161))
for details.

## ROS messages for Instance Stixels

The ROS message definitions can be found in a separate repository:
<https://gitlab.tudelft.nl/intelligent-vehicles/instance_stixels_msgs>
This allows to easily install and use the message definitions outside of the
singularity container.

# Installation without Singularity

Using Instance Stixels without our singularity image will require you to
install the necessary dependencies on your system, e.g. CUDA, TensorRT,
[a custom CUML fork](https://github.com/tomsal/cuml), PyTorch, etc.
This requires advanced knowledge and additional time, which is why we encourage
you to try our singularity image first.
To install all the required dependencies, we suggest to follow the steps
described in the `%post` section of the
[./singularity_recipe](./singularity_recipe).
It provides step-by-step commands, like a bash-script, you can modify according
to your wishes, e.g. ROS-specific parts are denoted in the comments.

# Acknowlegdement

Instance Stixels builds upon a variety of open source software. We would like
to thank the authors and contributors of the following projects for sharing
their code!
* GPU Stixels: https://github.com/dhernandez0/stixels
* Dilated Residual Network: https://github.com/fyu/drn
* Catch2: https://github.com/catchorg/Catch2
* RapidJSON: http://rapidjson.org/

# Contact

Please use the github issues and pull requests for code related questions.
For general questions, you find the email address of the corresponding author
in the papers and at http://www.intelligent-vehicles.org.
