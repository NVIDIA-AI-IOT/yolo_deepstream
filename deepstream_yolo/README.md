# Deploy YOLO Models With DeepStream #

**This sample shows how to integrate YOLO models with customized output layer parsing for detected objects with DeepStreamSDK.**

## 1. Sample contents: ##
- `deepstream_app_config_yolo.txt`: DeepStream reference app configuration file for using YOLO models as the primary detector.
- `config_infer_primary_yoloV4.txt`: Configuration file for the GStreamer nvinfer plugin for the YoloV4 detector model.
- `config_infer_primary_yoloV7.txt`: Configuration file for the GStreamer nvinfer plugin for the YoloV7 detector model.
- `nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp`: Output layer parsing function for detected objects for the Yolo models.

## 2. Pre-requisites: ##

### 2.1 Please make sure DeepStream 6.1.1+ is properly installed ###

### 2.2 Generate Model ###
#### YoloV4 

- Go to this pytorch repository <https://github.com/Tianxiaomo/pytorch-YOLOv4> where you can convert YOLOv4 Pytorch model into **ONNX**
- Other famous YOLOv4 pytorch repositories as references:
  - <https://github.com/WongKinYiu/PyTorch_YOLOv4>
  - <https://github.com/bubbliiiing/yolov4-pytorch>
  - <https://github.com/maudzung/Complex-YOLOv4-Pytorch>
  - <https://github.com/AllanYiin/YoloV4>
- Or you can download reference ONNX model directly from here ([link](https://drive.google.com/file/d/1tp1xzeey4YBSd8nGd-dkn8Ymii9ordEj/view?usp=sharing)).  

#### YOLOv7
following the guide https://github.com/WongKinYiu/yolov7#export, export a dynamic-batch-1-output onnx-model
```bash
$ python export.py --weights ./yolov7.pt --grid --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --dynamic-batch
```
or using the qat model exported from [yolov7_qat](../yolov7_qat)
## 3. Download and Run ##

```sh
  $ cd ~/
  $ git clone https://github.com/NVIDIA-AI-IOT/yolo_deepstream.git
  $ cd ~/yolo_deepstream/deepstream_yolo/nvdsinfer_custom_impl_Yolo
  $ make
  $ cd ..
```
  Make sure the model exists under ~/yolo_deepstream/deepstream_yolo/. Change the "config-file" parameter in the "deepstream_app_config_yolo.txt" configuration file to the nvinfer configuration file for the model you want to run with. 
|Model|Nvinfer Configuration File|
|-----------|----------|
|YoloV4|config_infer_primary_yoloV4.txt|
|YoloV7|config_infer_primary_yoloV7.txt|

```  
  $ deepstream-app -c deepstream_app_config_yolo.txt
```
## 4. CUDA Post Processing

this sample provide two ways of yolov7 post-processing(decoce yolo result, not include NMS), CPU version and GPU version
- CPU implement can be found in: [nvdsparsebbox_Yolo.cpp](deepstream_yolo/nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo.cpp)
- CUDA implement can be found in: [nvdsparsebbox_Yolo_cuda.cu](deepstream_yolo/nvdsinfer_custom_impl_Yolo/nvdsparsebbox_Yolo_cuda.cu)

Default will use CUDA-post processing. To enable CPU post-processing:
in [config_infer_primary_yoloV7.txt](deepstream_yolo/config_infer_primary_yoloV7.txt)

- `parse-bbox-func-name=NvDsInferParseCustomYoloV7_cuda` -> `parse-bbox-func-name=NvDsInferParseCustomYoloV7`
- `disable-output-host-copy=1` -> `disable-output-host-copy=0`

The performance of the CPU-post-processing and CUDA-post-processing result can be found in [Performance](https://github.com/NVIDIA-AI-IOT/yolo_deepstream#performance)

