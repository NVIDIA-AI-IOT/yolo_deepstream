# YOLOv4 and DeepStream #

**This sample shows how to integrate YOLOv4 with customized output layer parsing for detected objects with DeepStreamSDK.**

## 1. Sample contents: ##
- `deepstream_app_config_yolov4.txt`: DeepStream reference app configuration file for using YOLOv4 model as the primary detector.
- `config_infer_primary_yolov4.txt`: Configuration file for the GStreamer nvinfer plugin for the Yolo detector model.
- `nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp`: Output layer parsing function for detected objects for the Yolo model.

## 2. Pre-requisites: ##

### 2.1 Please make sure DeepStream 5.0 is properly installed ###

### 2.2 Generate YOLOv4 TensorRT Engine ###

- Go to this pytorch repository <https://github.com/Tianxiaomo/pytorch-YOLOv4> where you can convert YOLOv4 Pytorch model into **ONNX**
- Other famous YOLOv4 pytorch repositories as references:
  - <https://github.com/WongKinYiu/PyTorch_YOLOv4>
  - <https://github.com/bubbliiiing/yolov4-pytorch>
  - <https://github.com/maudzung/Complex-YOLOv4-Pytorch>
  - <https://github.com/AllanYiin/YoloV4>
- Or you can download reference ONNX model directly from here ([link](https://drive.google.com/file/d/1tp1xzeey4YBSd8nGd-dkn8Ymii9ordEj/view?usp=sharing)).  

## 3. Download and Run ##

```sh
  $ cd ~/
  $ git clone https://github.com/NVIDIA-AI-IOT/yolov4_deepstream.git
  $ cd ~/yolov4_deepstream/deepstream_yolov4/nvdsinfer_custom_impl_Yolo
  $ make
  $ cd ..
  // make sure model - yolov4_-1_3_416_416_nms_dynamic.onnx exists under deepstream_yolov4/, then run
  $ deepstream-app -c deepstream_app_config_yoloV4.txt
```
