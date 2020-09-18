/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! SampleYolo.cpp
//! This file contains the implementation of the YOLOv4 sample. It creates the network using
//! the YOLOv4 ONNX model.

#pragma once

#include "BatchStream.h"
#include "EntropyCalibrator.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvOnnxParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//!
//! \brief The SampleYoloParams structure groups the additional parameters required by
//!         the SSD sample.
//!
struct SampleYoloParams : public samplesCommon::OnnxSampleParams
{
    int outputClsSize = 80;              //!< The number of output classes
    int topK = 2000;
    int keepTopK = 1000;                   //!< The maximum number of detection post-NMS
    int nbCalBatches = 100;               //!< The number of batches for calibration
    int demo = 0;
    int speedTest = 0;
    int cocoTest = 0;
    size_t speedTestItrs = 1000;
    int explicitBatchSize = 1;
    std::vector<int> inputShape;
    std::vector<std::vector<int>> outputShapes;
    std::string inputImageName;
    std::string outputImageName;
    std::string calibrationBatches; //!< The path to calibration batches
    std::string engingFileName;
    std::string cocoClassNamesFileName;
    std::string cocoClassIDFileName;
    std::string cocoImageListFileName;
    std::string cocoImageOutputDir;
    std::string cocoTestResultFileName;
    std::string cocoImageDir;
};

struct BoundingBox
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int cls;
};

enum NMS_TYPE
{
    MIN,
    UNION,
};

struct SpeedInfo
{
    long long preProcess;
    long long model;
    long long postProcess;

    SpeedInfo() :
        preProcess {0},
        model {0},
        postProcess {0}
    {}

    void printTimeConsmued()
    {
        std::cout << "Time consumed in preProcess: " << this->preProcess << std::endl;
        std::cout << "Time consumed in model: " << this->model << std::endl;
        std::cout << "Time consumed in postProcess: " << this->postProcess << std::endl;
    }
};

class BoundingBoxComparator
{
public:
    bool operator() (const BoundingBox & b1, const BoundingBox & b2)
    {
        return b1.score > b2.score;
    }
};

class StringComparator
{
public:
    bool operator() (const std::string & first, const std::string & second) const
    { 
        return first < second; 
    }
};

//! \brief  The SampleYolo class implements the SSD sample
//!
//! \details It creates the network using a caffe model
//!
class SampleYolo
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    static const std::string gSampleName;
    
    SampleYolo(const SampleYoloParams& params);

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

    //!
    //! \brief Cleans up any state created in the sample class
    //!
    bool teardown();

private:
    SampleYoloParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.

    cv::Mat mSampleImage;

    SpeedInfo mSpeedInfo;

    //std::vector<samplesCommon::PPM<3, 320, 512>> mPPMs; //!< PPMs of test images

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    std::vector<std::string> mClasses;

    std::map<std::string, int, StringComparator> mClassesMap;

    std::vector<std::string> mImageFiles;

    std::ofstream mCocoResult;

    std::vector<int> image_rows;
    std::vector<int> image_cols;
    std::vector<int> image_pad_rows;
    std::vector<int> image_pad_cols;

    size_t mImageIdx;

    //!
    //! \brief Parses an ONNX model for YOLO and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
    //!
    bool processInput_aspectRatio(const samplesCommon::BufferManager& buffers);

    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Filters output detections and verify results
    //!
    bool verifyOutput_aspectRatio(const samplesCommon::BufferManager& buffers);

    bool verifyOutput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief To check if certain file exists given the path
    //!
    bool fileExists(const std::string& name)
    {
        std::ifstream f(name.c_str());
        return f.good();
    }

    bool infer_iteration(SampleUniquePtr<nvinfer1::IExecutionContext> &context, samplesCommon::BufferManager &buffers);

    std::vector<std::vector<BoundingBox>> get_bboxes(int batch_size, int keep_topk,
        int32_t *num_detections, float *mnsed_boxes, float *mnsed_scores, float *mnsed_classes);

    void draw_bboxes(const std::vector<BoundingBox> &bboxes, cv::Mat &img);

    void draw_coco_test_bboxes(const std::vector<BoundingBox> &bboxes, cv::Mat &img, int img_id);

    long long now_in_milliseconds();
};

