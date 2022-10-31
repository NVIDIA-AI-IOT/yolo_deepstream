/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"
#include "nvtx3/nvToolsExt.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

static const int NUM_CLASSES_YOLO = 80;
#define OBJECTLISTSIZE 25200
#define BLOCKSIZE  1024
thrust::device_vector<NvDsInferParseObjectInfo> objects_v(OBJECTLISTSIZE);

extern "C" bool NvDsInferParseCustomYoloV7_cuda(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);


__global__ void decodeYoloV7Tensor_cuda(NvDsInferParseObjectInfo *binfo/*output*/, float* data, int dimensions, int rows,
                                        int netW, int netH, float Threshold){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < rows) {
        data = data + idx * dimensions;
        float maxProb = data[ 4];
        //maxProb < Threshold, directly return
        if(maxProb < Threshold){
            binfo[idx].detectionConfidence = 0;
            return;
        }
        float bx = data[ 0];
        float by = data[ 1];
        float bw = data[ 2];
        float bh = data[ 3];
        int  maxIndex = data[ 5];
        float * classes_scores = data + 5;
        float maxScore = 0;
        int index = 0;

        #pragma unroll
        for (int j = 0 ;j < NUM_CLASSES_YOLO; j++){
           if(*classes_scores > maxScore){
              index = j;
              maxScore = *classes_scores;
           }
           classes_scores++;
        }
        maxIndex = index;
        float stride = 1.0;
        float xCenter = bx * stride;
        float yCenter = by * stride;
        float x0 = xCenter - bw / 2;
        float y0 = yCenter - bh / 2;
        float x1 = x0 + bw;
        float y1 = y0 + bh;
        x0 = fminf(float(netW), fmaxf(float(0.0), x0));
        y0 = fminf(float(netH), fmaxf(float(0.0), y0));
        x1 = fminf(float(netW), fmaxf(float(0.0), x1));
        y1 = fminf(float(netH), fmaxf(float(0.0), y1));
        binfo[idx].left = x0;
        binfo[idx].top = y0;
        binfo[idx].width = fminf(float(netW), fmaxf(float(0.0), x1-x0));
        binfo[idx].height = fminf(float(netH), fmaxf(float(0.0), y1-y0));
        binfo[idx].detectionConfidence = maxProb;
        binfo[idx].classId = maxIndex;
    }
    return;
}
static bool NvDsInferParseYoloV7_cuda(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
 
    if (outputLayersInfo.empty()) {
        std::cerr << "Could not find output layer in bbox parsing" << std::endl;;
        return false;
    }
    const NvDsInferLayerInfo &layer = outputLayersInfo[0];

    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }

    float* data = (float*)layer.buffer;
    const int dimensions = layer.inferDims.d[1];
    int rows = layer.inferDims.numElements / layer.inferDims.d[1];
    
    int GRIDSIZE = ((OBJECTLISTSIZE-1)/BLOCKSIZE)+1;
    //find the min threshold
    float min_PreclusterThreshold = *(std::min_element(detectionParams.perClassPreclusterThreshold.begin(),
        detectionParams.perClassPreclusterThreshold.end()));
    decodeYoloV7Tensor_cuda<<<GRIDSIZE,BLOCKSIZE>>>
        (thrust::raw_pointer_cast(objects_v.data()), data, dimensions, rows, networkInfo.width, 
        networkInfo.height, min_PreclusterThreshold);
    objectList.resize(OBJECTLISTSIZE);
    thrust::copy(objects_v.begin(),objects_v.end(),objectList.begin());//the same as cudamemcpy

    return true;
}

extern "C" bool NvDsInferParseCustomYoloV7_cuda(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    nvtxRangePush("NvDsInferParseYoloV7");
    bool ret = NvDsInferParseYoloV7_cuda (
        outputLayersInfo, networkInfo, detectionParams, objectList);

    nvtxRangePop();
    return ret;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV7_cuda);
