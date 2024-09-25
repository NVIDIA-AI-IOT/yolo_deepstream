# YoloV7 Quantization Aware Training
## Description
 We use [TensorRT's pytorch quntization tool](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization) to finetune training QAT yolov7 from the pre-trained weight, then export the model to onnx and deploy it with TensorRT. The accuray and performance can be found in below table.

|  Method   | Calibration method  | mAP<sup>val<br>0.5|mAP<sup>val<br>0.5:0.95 |batch-1 fps<br>Jetson Orin-X  |batch-16 fps<br>Jetson Orin-X  |weight|
|  ----  | ----  |----  |----  |----|----|-|
| pytorch FP16 | -             | 0.6972 | 0.5120 |-|-|[yolov7.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt)|
| pytorch PTQ-INT8  | Histogram(MSE)  | 0.6957 | 0.5100 |-|-|[yolov7_ptq.pt](https://nvidia.box.com/shared/static/j0rclm9k2ymj6ahdx55dxnnskzq91flh) [yolov7_ptq_640.onnx](https://nvidia.box.com/shared/static/rlv3buq7sei2log2d3beyg1jhjyw59hn)|
| pytorch QAT-INT8  | Histogram(MSE)  | 0.6961 | 0.5111 |-|-|[yolov7_qat.pt](https://nvidia.box.com/shared/static/vph9af9rbe7ed7ibfnajsk248mw9nq9f)|
| TensorRT FP16| -             | 0.6973 | 0.5124 |140 |168|[yolov7.onnx](https://nvidia.box.com/shared/static/rmh8rttesg4cgrysb2qm12udpvd95as1) |
| TensorRT PTQ-INT8 | TensorRT built in EntropyCalibratorV2 | 0.6317 | 0.4573 |207|264|-|
| TensorRT QAT-INT8 | Histogram(MSE)  | 0.6962 | 0.5113 |207|266|[yolov7_qat_640.onnx](https://nvidia.box.com/shared/static/v1ze885p35hfjl96xtw8s0xbcpv64tfr)|
 - network input resolution: 3x640x640
 - note: trtexec cudaGraph is enabled

## How To QAT Training
### 1.Setup

Suggest to use docker environment.
```bash
$ docker pull nvcr.io/nvidia/pytorch:22.09-py3
```

1. Clone and apply patch
```bash
# use this YoloV7 as a sample base 
git clone https://github.com/WongKinYiu/yolov7.git
cp -r yolov_deepstream/yolov7_qat/* yolov7/
```

2. Install dependencies
```bash
$ pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
```

3. Download dataset and pretrained model
```bash
$ bash scripts/get_coco.sh
$ wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

### 2. Start QAT training
  ```bash
  $ python scripts/qat.py quantize yolov7.pt --ptq=ptq.pt --qat=qat.pt --eval-ptq --eval-origin
  ```
  This script includes steps below: 
  - Insert Q&DQ nodes to get fake-quant pytorch model<br>
  [Pytorch quntization tool](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization) provides automatic insertion of QDQ function. But for yolov7 model, it can not get the same performance as PTQ, because in Explicit mode(QAT mode), TensorRT will henceforth refer Q/DQ nodes' placement to restrict the precision of the model. Some of the automatic added Q&DQ nodes can not be fused with other layers which will cause some extra useless precision convertion. In our script, We find Some rules and restrictions for yolov7, QDQ nodes are automatically analyzed and configured in a rule-based manner, ensuring that they are optimal under TensorRT. Ensuring that all nodes are running INT8(confirmed with tool:[trt-engine-explorer](https://github.com/NVIDIA/TensorRT/tree/main/tools/experimental/trt-engine-explorer), see [scripts/draw-engine.py](./scripts/draw-engine.py)). for details of this part, please refer [quantization/rules.py](./quantization/rules.py), About the guidance of Q&DQ insert, please refer [Guidance_of_QAT_performance_optimization](./doc/Guidance_of_QAT_performance_optimization.md)

  - PTQ calibration<br>
  After inserting Q&DQ nodes, we recommend to run PTQ-Calibration first. Per experiments, `Histogram(MSE)` is the best PTQ calibration method for yolov7.
  Note: if you are satisfied with PTQ result, you could also skip QAT.
  
  - QAT training<br>
  After QAT, need to finetune traning our model. after getting the accuracy we are satisfied, Saving the weights to files

### 3. Export onnx 
  ```bash
  $ python scripts/qat.py export qat.pt --size=640 --save=qat.onnx --dynamic
  ```

### 4. Evaluate model accuracy on coco 
  ```bash
  $ bash scripts/eval-trt.sh qat.pt
  ```

### 5. Benchmark
  ```bash
  $ /usr/src/tensorrt/bin/trtexec --onnx=qat.onnx --int8 --fp16  --workspace=1024000 --minShapes=images:4x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:4x3x640x640
  ```


## Quantization Yolov7-Tiny
```bash
$ python scripts/qat.py quantize yolov7-tiny.pt --qat=qat.pt --ptq=ptq.pt --ignore-policy="model\.77\.m\.(.*)|model\.0\.(.*)" --supervision-stride=1 --eval-ptq --eval-origin
```

## Note
- For YoloV5, please use the script `scripts/qat-yolov5.py`. This adds QAT support for `Add operator`, making it more performant.
- Please refer to the `quantize.replace_bottleneck_forward` function to handle the `Add operator`.
