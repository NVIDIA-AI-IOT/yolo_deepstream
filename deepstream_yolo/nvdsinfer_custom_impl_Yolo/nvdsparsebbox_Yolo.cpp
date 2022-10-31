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

static const int NUM_CLASSES_YOLO = 80;

float clamp(const float val, const float minVal, const float maxVal)
{
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

extern "C" bool NvDsInferParseCustomYoloV4(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseCustomYoloV7(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

/* YOLOv4 implementations */
static NvDsInferParseObjectInfo convertBBoxYoloV4(const float& bx1, const float& by1, const float& bx2,
                                     const float& by2, const uint& netW, const uint& netH)
{
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution

    float x1 = bx1 * netW;
    float y1 = by1 * netH;
    float x2 = bx2 * netW;
    float y2 = by2 * netH;

    x1 = clamp(x1, 0, netW);
    y1 = clamp(y1, 0, netH);
    x2 = clamp(x2, 0, netW);
    y2 = clamp(y2, 0, netH);

    b.left = x1;
    b.width = clamp(x2 - x1, 0, netW);
    b.top = y1;
    b.height = clamp(y2 - y1, 0, netH);

    return b;
}

static void addBBoxProposalYoloV4(const float bx, const float by, const float bw, const float bh,
                     const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBoxYoloV4(bx, by, bw, bh, netW, netH);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

static std::vector<NvDsInferParseObjectInfo>
decodeYoloV4Tensor(
    const float* boxes, const float* scores,
    const uint num_bboxes, NvDsInferParseDetectionParams const& detectionParams,
    const uint& netW, const uint& netH)
{
    std::vector<NvDsInferParseObjectInfo> binfo;

    uint bbox_location = 0;
    uint score_location = 0;
    for (uint b = 0; b < num_bboxes; ++b)
    {
        float bx1 = boxes[bbox_location];
        float by1 = boxes[bbox_location + 1];
        float bx2 = boxes[bbox_location + 2];
        float by2 = boxes[bbox_location + 3];

        float maxProb = 0.0f;
        int maxIndex = -1;

        for (uint c = 0; c < detectionParams.numClassesConfigured; ++c)
        {
            float prob = scores[score_location + c];
            if (prob > maxProb)
            {
                maxProb = prob;
                maxIndex = c;
            }
        }

        if (maxProb > detectionParams.perClassPreclusterThreshold[maxIndex])
        {
            addBBoxProposalYoloV4(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
        }

        bbox_location += 4;
        score_location += detectionParams.numClassesConfigured;
    }

    return binfo;
}

extern "C" bool NvDsInferParseCustomYoloV4(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
    }

    std::vector<NvDsInferParseObjectInfo> objects;

    const NvDsInferLayerInfo &boxes = outputLayersInfo[0]; // num_boxes x 4
    const NvDsInferLayerInfo &scores = outputLayersInfo[1]; // num_boxes x num_classes

    // 3 dimensional: [num_boxes, 1, 4]
    assert(boxes.inferDims.numDims == 3);
    // 2 dimensional: [num_boxes, num_classes]
    assert(scores.inferDims.numDims == 2);

    // The second dimension should be num_classes
    assert(detectionParams.numClassesConfigured == scores.inferDims.d[1]);
    
    uint num_bboxes = boxes.inferDims.d[0];

    // std::cout << "Network Info: " << networkInfo.height << "  " << networkInfo.width << std::endl;

    std::vector<NvDsInferParseObjectInfo> outObjs =
        decodeYoloV4Tensor(
            (const float*)(boxes.buffer), (const float*)(scores.buffer), num_bboxes, detectionParams,
            networkInfo.width, networkInfo.height);

    objects.insert(objects.end(), outObjs.begin(), outObjs.end());

    objectList = objects;

    return true;
}
/* YOLOv4 implementations end*/

/*Yolov7 bbox parser*/
static NvDsInferParseObjectInfo convertBBoxYoloV7(const float& bx, const float& by, const float& bw,
                                     const float& bh, const int& stride, const uint& netW,
                                     const uint& netH)
{
    NvDsInferParseObjectInfo b;
    // Restore coordinates to network input resolution
    float xCenter = bx * stride;
    float yCenter = by * stride;
    float x0 = xCenter - bw / 2;
    float y0 = yCenter - bh / 2;
    float x1 = x0 + bw;
    float y1 = y0 + bh;

    x0 = clamp(x0, 0, netW);
    y0 = clamp(y0, 0, netH);
    x1 = clamp(x1, 0, netW);
    y1 = clamp(y1, 0, netH);

    b.left = x0;
    b.width = clamp(x1 - x0, 0, netW);
    b.top = y0;
    b.height = clamp(y1 - y0, 0, netH);

    return b;
}

static void addBBoxProposalYoloV7(const float bx, const float by, const float bw, const float bh,
                     const uint stride, const uint& netW, const uint& netH, const int maxIndex,
                     const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
    NvDsInferParseObjectInfo bbi = convertBBoxYoloV7(bx, by, bw, bh, stride, netW, netH);
    if (bbi.width < 1 || bbi.height < 1) return;

    bbi.detectionConfidence = maxProb;
    bbi.classId = maxIndex;
    binfo.push_back(bbi);
}

static bool NvDsInferParseYoloV7(
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

    std::vector<NvDsInferParseObjectInfo> objects;

    float* data = (float*)layer.buffer;
    const int dimensions = layer.inferDims.d[1];
    int rows = layer.inferDims.numElements / layer.inferDims.d[1];

    for (int i = 0; i < rows; ++i) {
        //85 = x, y, w, h, maxProb, score0......score79
        float bx = data[ 0];
        float by = data[ 1];
        float bw = data[ 2];
        float bh = data[ 3];
        float maxProb = data[ 4];
        int  maxIndex = data[ 5];
        float * classes_scores = data + 5;
        
        float maxScore = 0;
        int index = 0;
        for (int j = 0 ;j < NUM_CLASSES_YOLO; j++){
           if(*classes_scores > maxScore){
              index = j;
              maxScore = *classes_scores;
           }
           classes_scores++;
        }
	
        maxIndex = index;
        data += dimensions;
        
        addBBoxProposalYoloV7(bx, by, bw, bh, 1, networkInfo.width, networkInfo.height, maxIndex, maxProb, objects);    
    }
    objectList = objects;
    return true;
}

extern "C" bool NvDsInferParseCustomYoloV7(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    return NvDsInferParseYoloV7 (
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV4);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV7);
