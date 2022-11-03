# yolo deepstream

##  Description

This repo have 4 parts:
### 1) yolov7_qat
In [yolov7_qat](yolov7_qat), We use [TensorRT's pytorch quntization tool](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization) to Finetune training QAT yolov7 from the pre-trained weight. 
Finally we get the same performance of PTQ in TensorRT on Jetson OrinX. And the accuracy(mAP) of the model only dropped a little.

### 2) tensorrt_yolov7
In [tensorrt_yolov7](tensorrt_yolov7), We provide a standalone c++ yolov7-app sample here. You can use trtexec to convert FP32 onnx models or QAT-int8 models exported from repo [yolov7_qat](yolov7_qat) to trt-engines. And set the trt-engine as yolov7-app's input. It can do detections on images/videos. Or test mAP on COCO dataset.

### 3) deepstream_yolo
In [deepstream_yolo](deepstream_yolo), This sample shows how to integrate YOLO models with customized output layer parsing for detected objects with DeepStreamSDK.

### 4) tensorrt_yolov4
In [tensorrt_yolov4](tensorrt_yolov4), This sample shows a standalone tensorrt-sample for yolov4.

## Performance
For YoloV7 sample:

Below table shows the end-to-end performance of processing 1080p videos with this sample application.
- Testing Device : 

  1. Jetson AGX Orin 64GB(PowerMode:MAXN + GPU-freq:1.3GHz + CPU:12-core-2.2GHz)

  2. Tesla T4

|Device      |precision      |Number <br>of streams | Batch Size | trtexec FPS| deepstream-app FPS<br>with cuda-post-process |deepstream-app FPS<br> with cpu-post-process|
|-----------    |-----------    |----------------- | -----------|----------- |-----------|-----------|
|  OrinX|  fp16         |  1               |     1      |       126  | 124       |   120     |
|  OrinX|  fp16         |  16              |    16      |       162  | 145       |   135     |
|  OrinX|  int8(PTQ/QAT)|  1               |     1      |       180  | 175       |   128      |
|  OrinX|  int8(PTQ/QAT)|  16              |    16      |       264  | 264       |   135      |
|  T4   |  fp16         |  1               |     1      |      132   |    125    |  123      |
|  T4   |  fp16         |  16              |    16      |      164   |   169     |   123     |
|  T4   |  int8(PTQ/QAT)|  1               |     1      |     208  | 133      |    127    |
|  T4   |  int8(PTQ/QAT)|  16              |    16      |     305    |  300      |   132      |


- note: trtexec cudaGraph not enabled as deepstream not support cudaGraph

## Code structure
```bash
├── deepstream_yolo
│   ├── config_infer_primary_yoloV4.txt # config file for yolov4 model
│   ├── config_infer_primary_yoloV7.txt # config file for yolov7 model
│   ├── deepstream_app_config_yolo.txt # deepStream reference app configuration file for using YOLOv models as the primary detector.
│   ├── labels.txt # labels for coco detection # output layer parsing function for detected objects for the Yolo model.
│   ├── nvdsinfer_custom_impl_Yolo 
│   │   ├── Makefile
│   │   └── nvdsparsebbox_Yolo.cpp 
│   └── README.md 
├── README.md
├── tensorrt_yolov4
│   ├── data 
│   │   ├── demo.jpg # the demo image
│   │   └── demo_out.jpg # image detection output of the demo image
│   ├── Makefile
│   ├── Makefile.config
│   ├── README.md
│   └── source
│       ├── generate_coco_image_list.py # python script to get list of image names from MS COCO annotation or information file
│       ├── main.cpp # program main entrance where parameters are configured here
│       ├── Makefile
│       ├── onnx_add_nms_plugin.py # python script to add BatchedNMSPlugin node into ONNX model
│       ├── SampleYolo.cpp # yolov4 inference class functions definition file
│       └── SampleYolo.hpp # yolov4 inference class definition file
├── tensorrt_yolov7
│   ├── CMakeLists.txt
│   ├── imgs # the demo images
│   │   ├── horses.jpg 
│   │   └── zidane.jpg
│   ├── README.md
│   ├── samples 
│   │   ├── detect.cpp # detection app for images detection
│   │   ├── validate_coco.cpp # validate coco dataset app
│   │   └── video_detect.cpp # detection app for video detection
│   ├── src
│   │   ├── argsParser.cpp # argsParser helper class for commandline parsing
│   │   ├── argsParser.h # argsParser helper class for commandline parsing
│   │   ├── tools.h # helper function for yolov7 class
│   │   ├── Yolov7.cpp # Class Yolov7
│   │   └── Yolov7.h # Class Yolov7
│   └── test_coco_map.py # tool for test coco map with json file
└── yolov7_qat
    ├── doc
    │   ├── Guidance_of_QAT_performance_optimization.md # guidance for Q&DQ insert and placement for pytorch-quantization tool
    ├── quantization
    │   ├── quantize.py # helper class for quantize yolov7 model
    │   └── rules.py # rules for Q&DQ nodes insert and restrictions
    ├── README.md 
    └── scripts
        ├── detect-trt.py # detect a image with tensorrt engine
        ├── draw-engine.py # draw tensorrt engine to graph
        ├── eval-trt.py # the script for evalating tensorrt mAP
        ├── eval-trt.sh # the command lne script for evaluating tensorrt mAP
        ├── qat.py # main function for QAT and PTQ
        └── trt-int8.py # tensorrt build-in calibration
```
