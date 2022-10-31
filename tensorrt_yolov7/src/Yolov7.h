
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


//!
//! This file contains the implementation of the yolov7 sample. 
//!
#pragma once

#include <iostream>
#include <string>
#include <memory>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include <vector>
#include <NvInferPlugin.h>
#include <tools.h>
#include <fstream>
#include <algorithm>
#include <numeric>
//opencv for preprocessing &  postprocessing
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Yolov7 {
public:
    //!
    //! \brief init Yolov7 class object
    //!
    //! \param engine_path The path of trt engine file
    //!
    Yolov7(std::string engine_path);

    //!
    //! \brief preprocess a list of image, the image will remembered inside the class by Yolov7 object
    //!
    //! \param cv_img  input images with BGR-UInt8, the size of the vector must smmaller than the maxBatchsize of the model
    //!
    std::vector<cv::Mat> preProcess(std::vector<cv::Mat> &cv_img);// 

    //!
    //! \brief run tensorRT inference with the data preProcessed
    //!
    int infer();

    //!
    //! \brief PostProcess, will decode and nms the batch inference result of yolov7
    //!
    //! \param cv_img  
    //! \return return all the nms result of Yolov7 
    //!
    std::vector<std::vector<std::vector<float>>> PostProcess(float iou_thres = 0.45f, float conf_thres = 0.25f);
    
    //!
    //! \brief Get the input dimenssion of the model
    //!
    //! \return return Dims of input
    //!
    nvinfer1::Dims getInputDim();

    //!
    //! \brief Get the output dimenssion of the model
    //!
    //! \return return the Dims of output
    //!
    nvinfer1::Dims getOutputDim();

    //!
    //! \brief Draw boxes on bgr image
    //! \param bgr_img The images need to be drawed with boxes
    //! \param nmsresult nms result get from PostProcess function
    //!
    static int Yolov7::DrawBoxesonGraph(cv::Mat &bgr_img, std::vector<std::vector<float>> nmsresult);

    //!
    //! \brief preprocess a list of image for validate mAP on coco dataset! the model must have a [batchsize, 3, 672, 672] input
    //!
    //! \param cv_img  input images with BGR-UInt8, the size of the vector must smmaller than the maxBatchsize of the model
    //!
    std::vector<cv::Mat> preProcess4Validate(std::vector<cv::Mat> &cv_img);

    //!
    //! \brief PostProcess for validate mAP on coco dataset!, will decode the batch inference result of yolov7
    //!
    //! \param cv_img  
    //! \return return all the nms result of Yolov7 
    //!
    std::vector<std::vector<std::vector<float>>> PostProcess4Validate(float iou_thres = 0.45f, float conf_thres = 0.25f);
private:

    int pushImg(void *imgBuffer, int numImg, bool fromCPU = true);

    std::vector<std::vector<std::vector<float>>> decode_yolov7_result(float conf_thres);
    std::vector<std::vector<std::vector<float>>> yolov7_nms(std::vector<std::vector<std::vector<float>>> &bboxes, float iou_thres);
    std::vector<std::vector<float>> nms(std::vector<std::vector<float>> &bboxes, float iou_thres);
    
    //TODO: to be imp
    void CudaGraphEndCapture(cudaStream_t stream);

    void CudaGraphBeginCapture(cudaStream_t stream);

    bool CudaGraphLaunch(cudaStream_t stream);

    bool enableCudaGraph();

    void ReportArgs();

private:

    int mImgPushed;
    int mMaxBatchSize;
    bool mDynamicBatch;

    //stream and event
    std::unique_ptr<CUstream_st, StreamDeleter> mStream;
    std::unique_ptr<CUevent_st, EventDeleter> mEvent;

    // trt objects
    std::unique_ptr<nvinfer1::IRuntime,TrtDeleter<nvinfer1::IRuntime>> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>> mContext;
    std::vector<std::unique_ptr<char, CuMemDeleter<char>>> mBindings;

    std::vector<void *> mBindingArray;
    std::vector<float> mHostOutputBuffer;
    std::vector<float> mHostNMSBuffer;

    std::string mEnginePath;
    nvinfer1::Dims mInputDim; //maxB,3,640,640
    nvinfer1::Dims mOutputDim;
    int mImgBufferSize;//sizeof(float)x3x640x640

    //cuda graph objects
    cudaGraph_t mGraph{};
    cudaGraphExec_t mGraphExec{};

    std::vector<std::vector<float>> md2i;

    bool mCudaGraphEnabled;

//TODOs
    //!
    //! get how many imgs has been totally processed
    //!
    // caculate fps real time
    unsigned long long mLast_inference_time;
    unsigned long long mTotal_inference_time;
    int mInference_count;
public:
    int imgProcessed() { return mInference_count; };
};
