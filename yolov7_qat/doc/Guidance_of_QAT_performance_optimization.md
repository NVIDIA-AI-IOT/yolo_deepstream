
# Get QAT models' best performance on TensorRT

## 1. Description
This guidance will show how to get the best performance QAT model on yolov7.

There are two workflows for quantizing networks in TensorRT, one is Post-training quantization (PTQ).(ref:[tensorrt-developer-guide/intro-quantization](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#intro-quantization)). The other is QAT.(ref:[tensorrt-developer-guide/work-with-qat-networks](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work-with-qat-networks). In PTQ mode, TensorRT will have the best performance, as it always choose the best layer fusion tactics and fastest kernels to make the global optimal network enqueue graph.
In QAT modes, the enqueue graph is designed by user. Which depends on the QDQ placement, The accuracy conversion and layer fusion strategies in the network are selected strictly according to the QDQ placement.(About the Q&DQ processing of TensorRT, please refer :[TensorRT-developer-guide: Processing of Q/DQ Networks](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#tensorrt-process-qdq)). That is, If we want to get the best performance of QAT, The Q&DQ nodes must make sure: 
1. All the computationally intensive layers will run with INT8.
2. Q&DQ can not break down the layer fusion of QAT model. 
3. Do not have unnecessary data conversion between INT8 and FLOAT

One effective way to get best performance of QAT is comparing the enqueue graph of QAT-TensorRT model with PTQ, and ensure they are the same.

## 2. Workflow
Our solution is: verbosing the QAT-Graph and compare with the PTQ-Graph. And back to fineTune the Q&DQ nodes placement. The procedure can be summaried as below.
1. Insert QDQ in the model and export it to onnx
2. Convert PTQ-Onnx and QAT-onnx to TensorRT model and draw the TensorRT-model-graph
3. Compare the TensorRT-enqueue-Graph and performance between QAT and PTQ
4. If the QAT Graph is different from PTQ Graph and the performance also wrose. modify the QDQ placement. Back to Step 1. Else, to Step 5
5. Run PTQ benchmark and QAT benchmark to verify

<img src="./imgs/QATFlow.png" width=50% alt="QATFlow" align=center />

For the layer-fusion rules: We can refer: [TensorRT-developer-guide: Types of Fusions](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#fusion-types)
For the tools for verbosing the TensorRT-model graph：[github-TensorRT: trt-engine-explorer](https://github.com/NVIDIA/TensorRT/tree/main/tools/experimental/trt-engine-explorer)(ref: [blog:exploring-tensorrt-engines-with-trex](https://developer.nvidia.com/blog/exploring-tensorrt-engines-with-trex/))


## 3. Step by step guidance of QAT optimization on yolov7

Now we will step by step optimizing a QAT model performance, We only care about the performance rather than accuracy at this time as we had not starting finetune the accuracy with training.
we use pytorch-quantization tool [pytorch-quantization](https://github.com/NVIDIA/TensorRT/blob/main/tools/pytorch-quantization) to quantize our pytorch model. And export onnx model with Q&DQ nodes.
This package provides a number of quantized layer modules, which contain quantizers for inputs and weights. e.g. `quant_nn.QuantLinear`, which can be used in place of `nn.Linear. ` These quantized layers can be substituted automatically, via monkey-patching, or by manually modifying the model definition.
Automatic layer substitution is done with `quant_modules`. This should be called before model creation.[ref: [pytorch-quantization-toolkit-tutorials](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/tutorials/quant_resnet50.html#quantizing-resnet50)]

### 1) Insert QDQ to model with monkey-patch quantization

with `quant_modules.initialize()` and `quant_modules.deactivate()`. The tool will automatic insert Q&DQ nodes in the network.

```python
quant_modules.initialize()
# Load PyTorch model
device = select_device(opt.device)
model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
labels = model.names
quant_modules.deactivate()
```
calibrate the onnx model to get the scale of Q&DQ nodes.
```python
def calibrate_model(model, model_name, data_loader, num_calib_batch, calibrator,hist_percentile, out_dir, device):
    """
        Feed data to the network and calibrate.
        Arguments:
            model: classification model
            model_name: name to use when creating state files
            data_loader: calibration data set
            num_calib_batch: amount of calibration passes to perform
            calibrator: type of calibration to use (max/histogram)
            hist_percentile: percentiles to be used for historgram calibration
            out_dir: dir to save state files in
    """
    if num_calib_batch > 0:
        print("Calibrating model")
        with torch.no_grad():
            collect_stats(model, data_loader, num_calib_batch, device)
        if not calibrator == "histogram":
            compute_amax(model, method="max")
            calib_output = os.path.join(
                out_dir,
                F"{model_name}-max-{num_calib_batch*data_loader.batch_size}.pth")
            ckpt = {'model': deepcopy(model)}
            torch.save(ckpt, calib_output)
        else:
            for percentile in hist_percentile:
                print(F"{percentile} percentile calibration")
                compute_amax(model, method="percentile")
                calib_output = os.path.join(
                    out_dir,
                    F"{model_name}-percentile-{percentile}-{num_calib_batch*data_loader.batch_size}.pth")
                ckpt = {'model': deepcopy(model)}
                torch.save(ckpt, calib_output)
            for method in ["mse", "entropy"]:
                print(F"{method} calibration")
                compute_amax(model, method=method)
                calib_output = os.path.join(
                    out_dir,
                    F"{model_name}-{method}-{num_calib_batch*data_loader.batch_size}.pth")
                ckpt = {'model': deepcopy(model)}
                torch.save(ckpt, calib_output)
```
### 2) export the calibrated-pytorch model to onnx
```python
quant_nn.TensorQuantizer.use_fb_fake_quant = True
torch.onnx.export(model, img, f, verbose=False, opset_version=13, input_names['images'],
                output_names=output_names,
                dynamic_axes=dynamic_axes)
quant_nn.TensorQuantizer.use_fb_fake_quant = False
```
***Now we got a onnx model with Q&DQ layers. TensorRT will process the onnx model with QDQ nodes as QAT models, With this way. Calibration is no longer needed as TensorRT will automatically performs INT8 quantization based on scales of Q and DQ nodes.***

TIPS: We calibrate the pytorch model with fake-quant, the exported onnx will have Q&DQ nodes. In the eye of pytorch, it is a ptq-model as we only did a calibration but no finetune training. But in the eye of TensorRT, as long as there are Q&DQ nodes inside the onnx， TensorRT will regard it as a QAT model.

### 3) Run TensorRT benchmark and export layers information to json
we can export the TensorRT-engine-graph and profile information with flag `--exportLayerInfo=layer.json --profilingVerbosity=detailed --exportProfile=profile.json`.
first we export fp32 onnx model
```bash
$ python export.py --weights ./yolov7.pt --grid --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
```
Then we copy the onnx to target device, Here we use Jetson OrinX as our target device, TensorRT has different behavior on different GPUs. So the test must run on your final target device

Run PTQ benchmark
```bash
$ /usr/src/tensorrt/bin/trtexec --onnx=yolov7.onnx --fp16 --int8 --verbose --saveEngine=yolov7_ptq.engine --workspace=1024000 --warmUp=500 --duration=10  --useCudaGraph --useSpinWait --noDataTransfers --exportLayerInfo=yolov7_ptq_layer.json --profilingVerbosity=detailed --exportProfile=yolov7_ptq_profile.json
```
Run fp16  benchmark
```bash
$ /usr/src/tensorrt/bin/trtexec --onnx=yolov7.onnx --fp16  --verbose --saveEngine=yolov7_fp16.engine --workspace=1024000 --warmUp=500 --duration=10  --useCudaGraph --useSpinWait --noDataTransfers --exportLayerInfo=yolov7_fp16_layer.json --profilingVerbosity=detailed --exportProfile=yolov7_fp16_profile.json
```
Run QAT benchmark
```bash
$ /usr/src/tensorrt/bin/trtexec --onnx=yolov7_qat.onnx --fp16 --int8 --verbose --saveEngine=yolov7_qat.engine --workspace=1024000 --warmUp=500 --duration=10  --useCudaGraph --useSpinWait --noDataTransfers --exportLayerInfo=yolov7_qat_layer.json --profilingVerbosity=detailed --exportProfile=yolov7_qat_profile.json
```

Run QAT_mask detect benchmark
```bash
$ /usr/src/tensorrt/bin/trtexec --onnx=yolov7_qat_maskdet.onnx --fp16 --int8 --verbose --saveEngine=yolov7_qat_maskdet.engine --workspace=1024000 --warmUp=500 --duration=10  --useCudaGraph --useSpinWait --noDataTransfers --exportLayerInfo=yolov7_qat_maskdet_layer.json --profilingVerbosity=detailed --exportProfile=yolov7_qat_maskdet_profile.json
```

We can get the fps from the log:
The PTQ performance is :
```bash
[I] Throughput: 206.562 qps
```
The fp16 performance is :
```bash
[I] Throughput: 139.597 qps
```
The version 1 QAT performance is:
```bash
[I] Throughput: 180.439 qps
```
That is not a good performance as we expect, Let's look insight the reason

### 4) Draw Engine graph

we use TensorRT opensource tool: [trt-engine-explorer](https://github.com/NVIDIA/TensorRT/tree/main/tools/experimental/trt-engine-explorer) drawing the enqueue graph of TensorRT. This tool take the trtexec exported layer json information as input.
Use the below code to draw the TensorRT-Engine-graph.(edit from `trt-engine-explorer/utils/draw_engine.py`)

```python
import graphviz
from trex import *
import argparse
import shutil


def draw_engine(engine_json_fname: str, engine_profile_fname: str):
    graphviz_is_installed =  shutil.which("dot") is not None
    if not graphviz_is_installed:
        print("graphviz is required but it is not installed.\n")
        print("To install on Ubuntu:")
        print("sudo apt --yes install graphviz")
        exit()

    plan = EnginePlan(engine_json_fname, engine_profile_fname)
    formatter = layer_type_formatter
    display_regions = True
    expand_layer_details = False

    graph = to_dot(plan, formatter,
                display_regions=display_regions,
                expand_layer_details=expand_layer_details)
    render_dot(graph, engine_json_fname, 'svg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', help="name of engine JSON file to draw")
    parser.add_argument('--profile', help="name of profile JSON file to draw")
    args = parser.parse_args()
    draw_engine(engine_json_fname=args.layer,engine_profile_fname=args.profile)
```
draw the graph:
```bash
$ python draw_engine.py --layer yolov7_qat_layer.json --profile yolov7_qat_profile.json
$ python draw_engine.py --layer yolov7_ptq_layer.json --profile yolov7_ptq_profile.json
```
we get `yolov7_qat_layer.json.svg` and `yolov7_ptq_layer.json.svg`

Let's see the difference:

<img src="./imgs/monkey-patch-qat-conv-fp16-issue_ptqonnx.png" width=200 alt="monkey-patch-qat-conv-fp16-issue_ptqonnx" align=center /><img src="./imgs/monkey-patch-qat-conv-fp16-issue_ptq.png" width=200 alt="monkey-patch-qat-conv-fp16-issue_ptq" align=center /><img src="./imgs/monkey-patch-qat-conv-fp16-issue_qatonnx.png" width=200 alt="monkey-patch-qat-conv-fp16-issue_qatonnx" align=center /><img src="./imgs/monkey-patch-qat-conv-fp16-issue_qat.png" width=200 alt="monkey-patch-qat-conv-fp16-issue_qatonnx" align=center />

- <center> pic1: The convolution layers before first concat layer in onnx </center>
- <center> pic2: pic1's TensorRT-graph </center>
- <center> pic3: the qat-onnx model </center>
- <center> pic4: pic3's TensorRt-graph </center>
- <center> (click to see full picture) </center>
 
### 5) Gap analyze and QDQ placement optimization
 There are a lot of useless int8->fp16 and fp16->int8 data convert in our QAT model. That is because : TensorRT will enforce the rules of QDQ to ensure consistent accuracy during inference and training(We didn't see any fp32 tensors here becasue TensorRT believes that fp16 will have the same accuracy as fp32)
 That is to say: If we want to reduce these useless data format convertion, We must edit our QDQ nodes to suit the fusion rules of TensorRT QAT.
 From the PTQ & QAT engine-graph, we can observed that: the concat layer will be reduced in TensorRT and all the input and output of concat will merge to one tensor(marked are red arrows in the below pic). If we do not guarantee the scale of Q&DQ nodes(marked with green circle in the below pic) of these tensors are the same. There will be redundant precision-conversion in our Graph.

   <img src="./imgs/monkey-patch-qat-conv-fp16-issue_qatonnx_edit.png" width=70% alt="monkey-patch-qat-conv-fp16-issue_qatonnx_edit" align=center />

For all the network-struct like this, We need do the same restrict. There is a special scene we need to take care: QDQ can cross some of the layers according to the commute rules from [TensorRT-developer-guide:tensorrt-process-qdq](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#tensorrt-process-qdq). eg. Max-pooling.
the DQ nodes marked with red circle will cross the MaxPool layer and TensorRT will remember the crossed-MaxPooling layer as int8 precision. Now we meet the similar scence as concat: We should restrict the scale of Q&DQ the same as the Q&DQ in the green circle to avoid generate  useless data format convertion here.

   <img src="./imgs/monkey-patch-qat-maxpooling-qat.png" width=50% alt="monkey-patch-qat-maxpooling-qat.png" align=center />

### 6) optimized QAT model's performance
Now we apply all the restriction we have metioned. We can test the performance:

we still use trtexec to benchmark the onnx model:
```bash
$ /usr/src/tensorrt/bin/trtexec --onnx=yolov7_qat_maskdet.onnx --fp16 --int8 --verbose --saveEngine=yolov7_qat_optimized.engine --workspace=1024000 --warmUp=500 --duration=10  --useCudaGraph --useSpinWait --noDataTransfers --exportLayerInfo=yolov7_qat_optimized_layer.json --profilingVerbosity=detailed --exportProfile=yolov7_qat_optimized_profile.json
[I] Throughput: 207.267 qps
```
This performance is almost the same as PTQ performance.

Next we need can finetune training our model to improve the accracy of the model.
