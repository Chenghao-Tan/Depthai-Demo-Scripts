# DepthAI Test Scripts (for boat obstacle avoidance)


## Overview
*(All configurable items are marked with #TODO.)*

1. **dataset**: MaSTr1325 dataset

2. **models**: Models trained with modified [DDRNet](https://github.com/Agent-Birkhoff/DDRNet) and UNet(deprecated, but still supported by the training framework). They are end-to-end, with built-in preprocessing.
    - Standard model's IO:
        - input->Image
        - output->confidence map of the obstacle pixels
    - WithDepth model's IO:
        - input->Image ("rgb"), Depth map ("depth")
        - output->flattened grids info ("out")
        - debug output->flattened grids info ("out"), filtered depth map ("debug")
    - Flattened grids info: (label, x, y, z for each grid, transmitted in 1D)
        - label: 0 for background, 1 for obstacles (binary classification for now)
        - x,y,z: in meters
    - Naming rules:
        - resolution (*W_H*)
        - resolution + grid number on each axis (*W_H_(GW_GH)*)

3. **tools**:
    - **device_manager**: Depthai's official tool for uploading bootloader, erasing flash, configuring, etc. (high DPI display bug fixed)
    - **get_info**: Get your camera's info, such as intrinsic matrices. You need to set RESOLUTION_H and RESOLUTION_W first to get corresponding intrinsic matrices.
    - **onnx_simplifier**: Used for simplifying onnx model. Run in CLI. The first arg is the path of the input onnx, and the second is the output.
    - json and txt: Cam info of my OAK-D-IOT-75. For reference.

4. **SpatialLocationCalculator**: See [Depthai's official demo of SpatialLocationCalculator](https://github.com/luxonis/depthai-experiments/tree/master/gen2-calc-spatials-on-host). The calculation is running on the host side. Use W/A/S/D to move, R to zoom out the ROI box, and F to zoom in.

5. **CamDemo**: Use RGB camera to test the obstacle segmentation model in real-time. It shows the original image and segmentation result in comparison. You can choose which blob to use (**path marked with #TODO**), but you may need to change the **IspScale** value if you do so. See [setIspScale](https://docs.luxonis.com/projects/api/en/latest/components/nodes/color_camera/#:~:text=setIspScale%28*,numerator%2C%20denominator%3E%20tuples). The scaled resolution must be bigger than the model's input. Using the same aspect ratio is highly recommended, or it will result in losing FOV.

6. **CamDemo(WithDepth)**: Similar with CamDemo, but using the WithDepth&debug version model. It shows the filtered depth map at 720P, along with the input image and depth map at their original size. There's an overlay on the filtered depth map showing the detection grids. You can also change the blob used. The related settings are the same as CamDemo. Please make sure **GRID_NUM_H** and **GRID_NUM_W** are compatible with the model. (It might be indicated by the name of the blob)

7. **DepthCrop**: Use the segmentation result to crop the depth map in real-time. The settings are the same as CamDemo(WithDepth).

8. **DepthTest**:

9. **XLinkDemo**:

10. **SendToESP**: Complete obstacle avoidance program, including the code for the onboard ESP32 and scripts for VPU. *This might be migrated to another repository soon.*
    - **vpu_setup**: Run in CLI. Used for uploading the program to VPU's flash. See the script detail for more usage.
    - **vpu_test**: Used for visualization like what CamDemo(WithDepth) does. **WILL NOT** upload the program to VPU's flash.
    - **main**: ESP32's main code. You need to build and flash it using ESP-IDF.
    - components and others: Libraries used in the project.
    - Configurations:
        - pipeline.py: **model path** and **IspScale**, the same as CamDemo(WithDepth).
        - vpu_test.py: **GRID_NUM_H**, **GRID_NUM_W** is the same as CamDemo(WithDepth). However, the model's **INPUT_SHAPE** needs to be specified manually. You can ignore these settings if you don't want to visualize.
        - main/config.h: **OBSTACLE_SEND_RATE_HZ** is the frequency of sending obstacle locations. **GRID_NUM**=GRID_NUM_H*GRID_NUM_H, is the total number of grids.


## Prepare

1. Install DepthAI.
    - Install python environment (It is recommended to use conda to install python3.9)
    - Clone [Depthai's official repository](https://github.com/luxonis/depthai) and follow the instruction there.

2. If you want to use the onboard ESP32, install ESP-IDF following the instruction [here](https://docs.espressif.com/projects/esp-idf/en/v4.2.2/esp32/get-started/index.html).

3. Clone this repository and its submodules:
``` bash
git clone https://github.com/Agent-Birkhoff/Depthai-Demo-Scripts.git
git submodule update --recursive --init
```
