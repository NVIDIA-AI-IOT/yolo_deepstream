# YOLOv4 and DeepStream #

**This sample shows how to integrate YOLOv4 with customized output layer parsing for detected objects with DeepStreamSDK.**

## 1. Sample contents: ##
- `deepstream_app_config_yolov4.txt`: DeepStream reference app configuration file for using YOLOv4 model as the primary detector.
- `config_infer_primary_yolov4.txt`: Configuration file for the GStreamer nvinfer plugin for the Yolo detector model.
- `nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp`: Output layer parsing function for detected objects for the Yolo model.

## 2. Pre-requisites: ##

### 2.1 Please make sure DeepStream 5.0 is properly installed ###

### 2.2 Generate YOLOv4 TensorRT Engine ###

- Go to this pytorch repository <https://github.com/Tianxiaomo/pytorch-YOLOv4> where you can convert YOLOv4 Pytorch model into **ONNX** and **TensorRT engine**

- Other famous YOLOv4 pytorch repositories as references:
  - <https://github.com/WongKinYiu/PyTorch_YOLOv4>
  - <https://github.com/bubbliiiing/yolov4-pytorch>
  - <https://github.com/maudzung/Complex-YOLOv4-Pytorch>
  - <https://github.com/AllanYiin/YoloV4>

## 3. Deploy TensorRT Engine into DeepStream ##

### 3.1 Copy this directory `deepstream_yolov4` into `<where_deepstream_is_installed>/deepstream-5.0/sources` ###

### 3.2 Compile `nvdsparsebbox_Yolo.cpp` in directory `nvdsinfer_custom_impl_Yolo` ###

```sh
  $ export CUDA_VER=<CUDA version of your environment, e.g. 10, 10.2, 11, etc>
  $ make
```

### 3.3 Copy the TensorRT engine into your working directory (`<where_deepstream_is_installed>/deepstream-5.0/sources/deepstream_yolov4`) ###

## 4. Update `config_infer_primary_yoloV4.txt` and `deepstream_app_config_yoloV4.txt` ##

### 4.1 Key options in `config_infer_primary_yoloV4.txt` that you may update for your own model ###

- [property]
  - model-engine-file
  - labelfile-path
  - batch-size
  - network-mode

- [class-attrs-all]
  - nms-iou-threshold
  - pre-cluster-threshold

Please refer to DeepStream plugin document for more information about plugin options

### 4.2 Key options in `deepstream_app_config_yoloV4.txt` that you may update for your own model ###

- [tiled-display]
  - width
  - height

- [streammux]
  - batch-size
  - width
  - height

- [primary-gie]
  - model-engine-file (the same as in [property])
  - labelfile-path (the same as in [property])
  - batch-size (the same as in [property])

- [tracker]
  - tracker-width
  - tracker-height

Please refer to DeepStream plugin document for more information about plugin options

## 5. Run the sample ##

```sh
  $ deepstream-app -c deepstream_app_config_yoloV4.txt
```
