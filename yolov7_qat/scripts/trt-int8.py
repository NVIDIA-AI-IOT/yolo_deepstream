################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import random
import cv2

# For ../common.py
import sys, os
TRT_LOGGER = trt.Logger()


def load_yolov7_coco_image(cocodir, topn = None):
    
    files = os.listdir(cocodir)
    files = [file for file in files if file.endswith(".jpg")]

    if topn is not None:
        np.random.seed(31)
        np.random.shuffle(files)
        files = files[:topn]

    datas = []

    # dataloader is setup pad=0.5
    for i, file in enumerate(files):
        if i == 0: continue
        if (i + 1) % 200 == 0:
            print(f"Load {i + 1} / {len(files)} ...")

        img = cv2.imread(os.path.join(cocodir, file))
        from_ = img.shape[1], img.shape[0]
        to_   = 640, 640
        scale = min(to_[0] / from_[0], to_[1] / from_[1])

        # low accuracy
        # M = np.array([
        #     [scale, 0, 16],
        #     [0, scale, 16],  # same to pytorch
        # ])

        # more accuracy
        M = np.array([
            [scale, 0, -scale * from_[0]  * 0.5  + to_[0] * 0.5 + scale * 0.5 - 0.5 + 16],
            [0, scale, -scale * from_[1] * 0.5 + to_[1] * 0.5 + scale * 0.5 - 0.5 + 16],  # same to pytorch
        ])
        input = cv2.warpAffine(img, M, (672, 672), borderValue=(114, 114, 114))
        input = input[..., ::-1].transpose(2, 0, 1)[None]   # BGR->RGB, HWC->CHW, CHW->1CHW
        input = (input / 255.0).astype(np.float32)
        datas.append(input)
        
    return np.concatenate(datas, axis=0)
    

class MNISTEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data, cache_file, batch_size=64):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        if not os.path.exists(cache_file):

            # Allocate enough memory for a whole batch.
            self.data = load_yolov7_coco_image(training_data, 1000)
            self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        batch = self.data[self.current_index : self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_int8_engine(onnx_file, calib, batch_size=32):
    with trt.Builder(
        TRT_LOGGER
    ) as builder, builder.create_network(1) as network, builder.create_builder_config() as config:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size.
        builder.max_batch_size = batch_size
        config.max_workspace_size = 1024 * 1024 * 1024  # 1024 MB
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calib
        with trt.OnnxParser(network, TRT_LOGGER) as parser:
            parser.parse_from_file(onnx_file)
        # network.mark_output(model_tensors.find(ModelData.OUTPUT_NAME))
        # Build engine and do int8 calibration.
        plan = builder.build_serialized_network(network, config)
        return bytes(plan)


def replace_suffix(file, new_suffix):
    r = file.rfind(".")
    return f"{file[:r]}{new_suffix}"


def main():
    # Now we create a calibrator and give it the location of our calibration data.
    # We also allow it to cache calibration data for faster engine building.
    onnxfile          = "yolov7.onnx"
    calibration_cache = replace_suffix(onnxfile, ".cache")
    engine_file       = replace_suffix(onnxfile, ".engine")
    calib = MNISTEntropyCalibrator("/datav/dataset/coco/images/train2017/", cache_file=calibration_cache)

    # Inference batch size can be different from calibration batch size.
    batch_size = 1
    engine_data = build_int8_engine(onnxfile, calib, batch_size)

    with open(engine_file, "wb") as f:
        f.write(engine_data)


if __name__ == "__main__":
    main()
