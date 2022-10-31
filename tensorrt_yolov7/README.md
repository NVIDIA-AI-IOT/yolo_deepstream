# Yolov7 TensorRT cpp

## Description
This is a yolov7 TensorRT cpp app. Fisrt, using trtexec to convert onnx model to FP32 or FP16 TensorRT engine ,or INT8 TensorRT engine from the QAT model finetuned from [yolov7_qat](./yolov7_qat).
Then you can use the `detect/video_detect` app to detect a list of images(images number must smaller than the batchsize of the model)/video. or use `validate_coco` app to test mAP of the TensorRT engine.
## Prerequisites
#### Install opencv
- Note: There are OpenCV4 dependencies in this program. 
Follow README and documents of this repository https://github.com/opencv/opencv to install OpenCV.
And, if you want use detect_video app, please install opencv with `ffmpeg` enabled

#### Install jsoncpp libs
jsoncpp lib is used to write coco-dataset-validate-result to json file. 
```bash
$ sudo apt-get install libjsoncpp-dev
```
## Build and Run yolov7-TensorRT-app
### Build
```bash
$ mkdir build && cd build
$ cmake ..
$ make -j4
```

### Prepare TensorRT engines

convert onnx model to tensorrt-engine
```bash
# fp32 model
$ /usr/src/tensorrt/bin/trtexec --onnx=yolov7.onnx --saveEngine=yolov7fp32.engine
# fp16 model
$ /usr/src/tensorrt/bin/trtexec --onnx=yolov7.onnx --saveEngine=yolov7fp16.engine --fp16
# int8 QAT model, the onnx model with Q&DQ nodes
$ /usr/src/tensorrt/bin/trtexec --onnx=yolov7qat.onnx --saveEngine=yolov7QAT.engine --fp16 --int8
```
### Detection & Validate
- detect with image:
    ```bash
    $ ./build/detect --engine=yolov7db4fp32.engine --img=./imgs/horses.jpg,./imgs/zidane.jpg
    ```
- detect with video:
    - note: only support batchsize = 1 now.
    ```bash
    $ ./build/video_detect --engine=./yolov7fp32.engine --video=YOUR_VIDEO_PATH.mp4
    ```
- validate mAP on dataset
    - note: validate_coco only support model inputsize `[batchsize, 3, 672, 672]`
    ```bash
    $ ./build/validate_coco --engine=./yolov7fp32.engine --coco=/YOUR/COCO/DATA/PATH/
    --------------------------------------------------------
    Yolov7 initialized from: yolov7672.engine
    input : images , shape : [ 1,3,672,672,]
    output : output , shape : [ 1,27783,85,]
    --------------------------------------------------------
    5000 / 5000
    predict result has been written to ./predict.json

    $ python test_coco_map.py --predict ./predict.json --coco /YOUR/COCO/DATA/PATH/
    ...
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.51005
    ...
    ```
