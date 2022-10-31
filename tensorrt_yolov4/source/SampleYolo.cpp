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
//! SampleYolo.cpp
//! This file contains the implementation of the YOYOv4 sample. It creates the network using
//! the YOLOV4 ONNX model.
//!

#include "SampleYolo.hpp"

#include <chrono>

const std::string SampleYolo::gSampleName = "TensorRT.sample_yolo";

int calculate_num_boxes(int input_h, int input_w)
{
    int num_anchors = 3;

    int h1 = input_h / 8;
    int h2 = input_h / 16;
    int h3 = input_h / 32;

    int w1 = input_w / 8;
    int w2 = input_w / 16;
    int w3 = input_w / 32;

    return num_anchors * (h1 * w1 + h2 * w2 + h3 * w3);
}


long long SampleYolo::now_in_milliseconds()
{
    return std::chrono::duration_cast <std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();  
}


SampleYolo::SampleYolo(const SampleYoloParams& params)
        : 
        mParams(params),
        mEngine(nullptr),
        mCocoResult(this->mParams.cocoTestResultFileName, std::ofstream::out),
        mImageIdx(0)
{
    char str[100];

    std::ifstream coco_names(this->mParams.cocoClassNamesFileName);  
    while(coco_names.getline(str, 100) )
    {
        std::string cls_name {str};
        this->mClasses.push_back(cls_name.substr(0, cls_name.size()));
    }
    coco_names.close();

    std::ifstream coco_categories(this->mParams.cocoClassIDFileName);
    while(coco_categories.getline(str, 100))
    {
        std::string id_and_name {str};
        auto id_str = id_and_name.substr(0, id_and_name.find("\t"));
        auto class_name = id_and_name.substr(id_and_name.find("\t") + 1, id_and_name.size());
        int class_id = std::stoi(id_str);

        this->mClassesMap[class_name] = class_id;
    }
    coco_categories.close();

    std::ifstream coco_image_list_file(this->mParams.cocoImageListFileName);
    while(coco_image_list_file.getline(str, 100))
    {
        this->mImageFiles.push_back(params.cocoImageDir + std::string("/") + std::string(str));
    }
    coco_image_list_file.close();

    std::cout << "There are " << this->mImageFiles.size() << " coco images to process" << std::endl;

    // Print json file header
    this->mCocoResult << "[" << std::endl;

    this->mSampleImage = cv::imread(this->mParams.inputImageName, /*CV_LOAD_IMAGE_COLOR*/-1);
}

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the YOLO network by parsing the ONNX model and builds
//!          the engine that will be used to run YOLO (this->mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleYolo::build()
{
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");

    if (this->fileExists(mParams.engingFileName))
    {
        std::vector<char> trtModelStream;
        size_t size{0};
        std::ifstream file(mParams.engingFileName, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream.resize(size);
            file.read(trtModelStream.data(), size);
            file.close();
        }

        IRuntime* infer = nvinfer1::createInferRuntime(sample::gLogger);
        if (mParams.dlaCore >= 0)
        {
            infer->setDLACore(mParams.dlaCore);
        }
        this->mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            infer->deserializeCudaEngine(trtModelStream.data(), size, nullptr), samplesCommon::InferDeleter());

        infer->destroy();

        sample::gLogInfo << "TRT Engine loaded from: " << mParams.engingFileName << std::endl;
        if (!this->mEngine)
        {
            return false;
        }
        else
        {
            this->mInputDims.nbDims = this->mParams.inputShape.size();
            this->mInputDims.d[0] = this->mParams.inputShape[0];
            this->mInputDims.d[1] = this->mParams.inputShape[1];
            this->mInputDims.d[2] = this->mParams.inputShape[2];
            this->mInputDims.d[3] = this->mParams.inputShape[3];

            return true;
        }
    }
    else
    {
        auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
        if (!builder)
        {
            return false;
        }

        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if (!network)
        {
            return false;
        }

        auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config)
        {
            return false;
        }

        auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
        if (!parser)
        {
            return false;
        }

        auto constructed = constructNetwork(builder, network, config, parser);
        if (!constructed)
        {
            return false;
        }

        assert(network->getNbInputs() == 1);
        this->mInputDims = network->getInput(0)->getDimensions();
        std::cout << this->mInputDims.nbDims << std::endl;
        assert(this->mInputDims.nbDims == 4);
    }

    return true;
}

//!
//! \brief Uses an onnx parser to create the YOLO Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the YOLO network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleYolo::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    // Parse ONNX model file to populate TensorRT INetwork
    int verbosity = (int) nvinfer1::ILogger::Severity::kERROR;

    sample::gLogInfo << "Parsing ONNX file: " << mParams.onnxFileName << std::endl;

    if (!parser->parseFromFile(mParams.onnxFileName.c_str(), verbosity))
    {
        sample::gLogError << "Unable to parse ONNX model file: " << mParams.onnxFileName << std::endl;
        return false;
    }

    builder->setMaxBatchSize(mParams.batchSize);

    config->setMaxWorkspaceSize(4096_MiB);

    config->setFlag(BuilderFlag::kGPU_FALLBACK);

    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    
    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;

    // issue for int8 mode
    if (mParams.int8)
    {
        BatchStream calibrationStream(
            mParams.explicitBatchSize, mParams.nbCalBatches, mParams.calibrationBatches, mParams.dataDirs);
        calibrator.reset(
            new Int8EntropyCalibrator2<BatchStream>(calibrationStream, 0, "Yolo", mParams.inputTensorNames[0].c_str()));
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
    }

    // Enable DLA if mParams.dlaCore is true
    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    sample::gLogInfo << "Building TensorRT engine" << mParams.engingFileName << std::endl;

    this->mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    
    if (!this->mEngine)
    {
        return false;
    }

    if (mParams.engingFileName.size() > 0)
    {
        std::ofstream p(mParams.engingFileName, std::ios::binary);
        if (!p)
        {
            return false;
        }
        nvinfer1::IHostMemory* ptr = this->mEngine->serialize();
        assert(ptr);
        p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
        ptr->destroy();
        p.close();
        sample::gLogInfo << "TRT Engine file saved to: " << mParams.engingFileName << std::endl;
    }

    return true;
}

bool SampleYolo::infer_iteration(SampleUniquePtr<nvinfer1::IExecutionContext> &context, samplesCommon::BufferManager &buffers)
{
    auto time1 = this->now_in_milliseconds();
    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput_aspectRatio(buffers))
    {
        return false;
    }

    auto time2 = this->now_in_milliseconds();

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());

    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    auto time3 = this->now_in_milliseconds();

    // Post-process detections and verify results
    if (!verifyOutput_aspectRatio(buffers))
    {
        return false;
    }

    auto time4 = this->now_in_milliseconds();

    this->mSpeedInfo.preProcess += time2 - time1;
    this->mSpeedInfo.model += time3 - time2;
    this->mSpeedInfo.postProcess += time4 - time3;

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleYolo::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(this->mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(this->mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    if (this->mParams.cocoTest)
    {
        auto limit = this->mImageFiles.size() / this->mParams.explicitBatchSize + 1;
        for (size_t i = 0; i < limit; ++i)
        {
            std::cout << "Iteration " << i << std::endl;
            this->infer_iteration(context, buffers);
        }
    }
    else if (this->mParams.speedTest)
    {
        auto limit = this->mParams.speedTestItrs;
        for (size_t i = 0; i < limit; ++i)
        {
            std::cout << "Iteration " << i << std::endl;
            this->infer_iteration(context, buffers);
        }
    }
    else
    {
        this->infer_iteration(context, buffers);
    }

    this->mSpeedInfo.printTimeConsmued();

    return true;
}

//!
//! \brief Cleans up any state created in the sample class
//!
bool SampleYolo::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    //nvcaffeparser1::shutdownProtobufLibrary();
    // nvonnxparser::
    this->mCocoResult << "]";
    this->mCocoResult.close();

    return true;
}

//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
bool SampleYolo::processInput_aspectRatio(const samplesCommon::BufferManager& buffers)
{
    const int inputB = this->mInputDims.d[0];
    const int inputC = this->mInputDims.d[1];
    const int inputH = this->mInputDims.d[2];
    const int inputW = this->mInputDims.d[3];

    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer(this->mParams.inputTensorNames[0]));

    // std::cout << inputC << " " << inputH << " " << inputW << std::endl;
    std::vector<std::vector<cv::Mat>> input_channels;
    for (int b = 0; b < inputB; ++b)
    {
        input_channels.push_back(std::vector<cv::Mat> {static_cast<size_t>(inputC)});
    }

    this->image_rows.clear();
    this->image_cols.clear();
    this->image_pad_rows.clear();
    this->image_pad_cols.clear();

    if (this->mParams.cocoTest)
    {
        for (int b = 0; b < inputB; ++b)
        {
            if (this->mImageIdx + b < this->mImageFiles.size())
            {
                cv::Mat test_img = cv::imread(this->mImageFiles[this->mImageIdx + b]);
                cv::Mat rgb_img;
                cv::cvtColor(test_img, rgb_img, cv::COLOR_BGR2RGB);
                cv::Mat pad_dst;
                cv::Scalar value(0, 0, 0);
                auto scaleSize = cv::Size(inputW, inputH);

                int rows = test_img.rows;
                int cols = test_img.cols;

                int pad_rows, pad_cols;

                if (inputH * cols < inputW * rows)
                {
                    // Add padding to cols
                    pad_rows = rows;
                    pad_cols = rows * inputW / inputH;
                    copyMakeBorder( rgb_img, pad_dst, 0, 0, (pad_cols - cols) / 2, (pad_cols - cols) / 2, cv::BORDER_CONSTANT, value );
                }
                else
                {
                    // Add padding to rows
                    pad_rows = cols * inputH / inputW;
                    pad_cols = cols;
                    copyMakeBorder( rgb_img, pad_dst, (pad_rows - rows) / 2, (pad_rows - rows) / 2, 0, 0, cv::BORDER_CONSTANT, value );
                }

                this->image_rows.push_back(rows);
                this->image_cols.push_back(cols);
                this->image_pad_rows.push_back(pad_rows);
                this->image_pad_cols.push_back(pad_cols);
                
                cv::Mat resized;
                cv::resize(pad_dst, resized, scaleSize, 0, 0, cv::INTER_LINEAR);
                cv::split(resized, input_channels[b]);
            }
            else
            {
                auto scaleSize = cv::Size(inputW, inputH);
                cv::Mat zeros = cv::Mat::zeros(scaleSize, CV_8UC3);
                cv::split(zeros, input_channels[b]);
            }
        }
    }
    else
    {
        cv::Mat rgb_img;

        // Convert BGR to RGB
        cv::cvtColor(this->mSampleImage, rgb_img, cv::COLOR_BGR2RGB);

        auto scaleSize = cv::Size(inputW, inputH);
        cv::Mat resized;
        cv::resize(rgb_img, resized, scaleSize, 0, 0, cv::INTER_LINEAR);
        
        // Each element in batch share the same image matrix
        for (int b = 0; b < inputB; ++b)
        {
            cv::split(resized, input_channels[b]);
        }
    }

    int volBatch = inputC * inputH * inputW;
    int volChannel = inputH * inputW;
    int volW = inputW;

    int d_batch_pos = 0;
    for (int b = 0; b < inputB; b++)
    {  
        int d_c_pos = d_batch_pos;
        for (int c = 0; c < inputC; c++)
        {
            int s_h_pos = 0;
            int d_h_pos = d_c_pos;
            for (int h = 0; h < inputH; h++)
            {
                int s_pos = s_h_pos;
                int d_pos = d_h_pos;
                for (int w = 0; w < inputW; w++)
                {
                    hostInputBuffer[d_pos] = (float)input_channels[b][c].data[s_pos] / 255.0f;
                    ++s_pos;
                    ++d_pos;
                }
                s_h_pos += volW;
                d_h_pos += volW;
            }
            d_c_pos += volChannel;
        }
        d_batch_pos += volBatch;
    }

    return true;
}


//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
bool SampleYolo::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputB = this->mInputDims.d[0];
    const int inputC = this->mInputDims.d[1];
    const int inputH = this->mInputDims.d[2];
    const int inputW = this->mInputDims.d[3];

    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer(this->mParams.inputTensorNames[0]));

    // std::cout << inputC << " " << inputH << " " << inputW << std::endl;
    std::vector<std::vector<cv::Mat>> input_channels;
    for (int b = 0; b < inputB; ++b)
    {
        input_channels.push_back(std::vector<cv::Mat> {static_cast<size_t>(inputC)});
    }

    this->image_rows.clear();
    this->image_cols.clear();
    this->image_pad_rows.clear();
    this->image_pad_cols.clear();

    if (this->mParams.cocoTest)
    {
        for (int b = 0; b < inputB; ++b)
        {
            if (this->mImageIdx + b < this->mImageFiles.size())
            {
                cv::Mat test_img = cv::imread(this->mImageFiles[this->mImageIdx + b]);
                cv::Mat rgb_img;
                cv::cvtColor(test_img, rgb_img, cv::COLOR_BGR2RGB);

                auto scaleSize = cv::Size(inputW, inputH);
                
                this->image_rows.push_back(test_img.rows);
                this->image_cols.push_back(test_img.cols);
                
                cv::Mat resized;
                cv::resize(rgb_img, resized, scaleSize, 0, 0, cv::INTER_LINEAR);
                cv::split(resized, input_channels[b]);
            }
            else
            {
                auto scaleSize = cv::Size(inputW, inputH);
                cv::Mat zeros = cv::Mat::zeros(scaleSize, CV_8UC3);
                cv::split(zeros, input_channels[b]);
            }
        }
    }
    else
    {
        cv::Mat rgb_img;

        // Convert BGR to RGB
        cv::cvtColor(this->mSampleImage, rgb_img, cv::COLOR_BGR2RGB);

        auto scaleSize = cv::Size(inputW, inputH);
        cv::Mat resized;
        cv::resize(rgb_img, resized, scaleSize, 0, 0, cv::INTER_LINEAR);
        
        // Each element in batch share the same image matrix
        for (int b = 0; b < inputB; ++b)
        {
            cv::split(resized, input_channels[b]);
        }
    }

    int volBatch = inputC * inputH * inputW;
    int volChannel = inputH * inputW;
    int volW = inputW;

    int d_batch_pos = 0;
    for (int b = 0; b < inputB; b++)
    {  
        int d_c_pos = d_batch_pos;
        for (int c = 0; c < inputC; c++)
        {
            int s_h_pos = 0;
            int d_h_pos = d_c_pos;
            for (int h = 0; h < inputH; h++)
            {
                int s_pos = s_h_pos;
                int d_pos = d_h_pos;
                for (int w = 0; w < inputW; w++)
                {
                    hostInputBuffer[d_pos] = (float)input_channels[b][c].data[s_pos] / 255.0f;
                    ++s_pos;
                    ++d_pos;
                }
                s_h_pos += volW;
                d_h_pos += volW;
            }
            d_c_pos += volChannel;
        }
        d_batch_pos += volBatch;
    }

    return true;
}

//!
//! \brief Filters output detections and verify result
//!
//! \return whether the detection output matches expectations
//!
bool SampleYolo::verifyOutput_aspectRatio(const samplesCommon::BufferManager& buffers)
{
    const int keepTopK = mParams.keepTopK;

    int32_t *num_detections = static_cast<int32_t*>(buffers.getHostBuffer(this->mParams.outputTensorNames[0]));
    float *nmsed_boxes = static_cast<float*>(buffers.getHostBuffer(this->mParams.outputTensorNames[1]));
    float *nmsed_scores = static_cast<float*>(buffers.getHostBuffer(this->mParams.outputTensorNames[2]));
    float *nmsed_classes = static_cast<float*>(buffers.getHostBuffer(this->mParams.outputTensorNames[3]));

    if (!num_detections || !nmsed_boxes || !nmsed_scores || !nmsed_classes)
    {
        std::cout << "NULL value output detected!" << std::endl;
    }

    auto nms_bboxes = this->get_bboxes(this->mParams.outputShapes[0][0], keepTopK, num_detections, nmsed_boxes, nmsed_scores, nmsed_classes);

    std::cout << "batch size: " << nms_bboxes.size() << std::endl;

    if (this->mParams.cocoTest)
    {
        for (int b = 0; b < this->mParams.explicitBatchSize; ++b)
        {
            if (this->mImageIdx + b >= this->mImageFiles.size())
            {
                break;
            }

            auto &bboxes_each_image = nms_bboxes[b];

            std::cout << "------------ Next Image! --------------" << std::endl;
            std::cout << "Number of detections: " << num_detections[b] << std::endl;

            std::string img_id_str = this->mImageFiles[this->mImageIdx + b].substr (this->mImageFiles[this->mImageIdx + b].size() - 16, 12);
            int img_id = std::stoi(img_id_str);

            for (size_t i = 0; i < bboxes_each_image.size(); ++i)
            {
                auto &bbox = bboxes_each_image[i];

                auto class_name = this->mClasses[bbox.cls];
                
                float x1 = bbox.x1 * this->image_pad_cols[b];
                float y1 = bbox.y1 * this->image_pad_rows[b];
                float x2 = bbox.x2 * this->image_pad_cols[b];
                float y2 = bbox.y2 * this->image_pad_rows[b];

                int x_off = (this->image_pad_cols[b] - this->image_cols[b]) / 2;
                int y_off = (this->image_pad_rows[b] - this->image_rows[b]) / 2;

                bbox.x1 = std::max(0.0f, x1 - x_off);
                bbox.y1 = std::max(0.0f, y1 - y_off);
                bbox.x2 = std::max(0.0f, x2 - x_off);
                bbox.y2 = std::max(0.0f, y2 - y_off);

                this->mCocoResult << std::fixed << std::setprecision(2);
                this->mCocoResult << "{\"image_id\":" << img_id << ",\"category_id\":" << this->mClassesMap[class_name] << ",\"bbox\":[";
                this->mCocoResult << bbox.x1 << "," << bbox.y1 << "," << bbox.x2 - bbox.x1 << "," << bbox.y2 - bbox.y1;
                this->mCocoResult << std::fixed << std::setprecision(4);
                this->mCocoResult << "],\"score\":" << bbox.score;
                
                if (i == bboxes_each_image.size() - 1 && this->mImageIdx + b == this->mImageFiles.size() - 1)
                {
                    this->mCocoResult << "}\n";
                }
                else
                {
                    this->mCocoResult << "},\n";
                }
            }
        }

        this->mImageIdx += this->mParams.explicitBatchSize;
    }
    else if (this->mParams.demo)
    {
        for (int b = 0; b < this->mParams.explicitBatchSize; ++b)
        {
            auto &bboxes_each_image = nms_bboxes[b];
            std::cout << "------------ Next Image! --------------" << std::endl;
            std::cout << "Number of detections: " << num_detections[b] << std::endl;

            for(auto &bbox : bboxes_each_image)
            {
                std::cout << "[ " << bbox.x1 << " " << bbox.y1 << " " << bbox.x2 << " " << bbox.y2 << " ] score: " << bbox.score << " class: " << bbox.cls << std::endl;
            }
        }

        // Draw bboxes only for the first image in each batch
        cv::Mat bgr_img_cpy = this->mSampleImage.clone();
        this->draw_bboxes(nms_bboxes[0], bgr_img_cpy);
    }

    bool pass = true;

    return pass;
}

//!
//! \brief Filters output detections and verify result
//!
//! \return whether the detection output matches expectations
//!
bool SampleYolo::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int keepTopK = mParams.keepTopK;

    int32_t *num_detections = static_cast<int32_t*>(buffers.getHostBuffer(this->mParams.outputTensorNames[0]));
    float *nmsed_boxes = static_cast<float*>(buffers.getHostBuffer(this->mParams.outputTensorNames[1]));
    float *nmsed_scores = static_cast<float*>(buffers.getHostBuffer(this->mParams.outputTensorNames[2]));
    float *nmsed_classes = static_cast<float*>(buffers.getHostBuffer(this->mParams.outputTensorNames[3]));

    if (!num_detections || !nmsed_boxes || !nmsed_scores || !nmsed_classes)
    {
        std::cout << "NULL value output detected!" << std::endl;
    }

    auto nms_bboxes = this->get_bboxes(this->mParams.outputShapes[0][0], keepTopK, num_detections, nmsed_boxes, nmsed_scores, nmsed_classes);

    std::cout << "batch size: " << nms_bboxes.size() << std::endl;

    if (this->mParams.cocoTest)
    {
        for (int b = 0; b < this->mParams.explicitBatchSize; ++b)
        {
            if (this->mImageIdx + b >= this->mImageFiles.size())
            {
                break;
            }

            auto &bboxes_each_image = nms_bboxes[b];

            std::cout << "------------ Next Image! --------------" << std::endl;
            std::cout << "Number of detections: " << num_detections[b] << std::endl;

            std::string img_id_str = this->mImageFiles[this->mImageIdx + b].substr (this->mImageFiles[this->mImageIdx + b].size() - 16, 12);
            int img_id = std::stoi(img_id_str);

            for (size_t i = 0; i < bboxes_each_image.size(); ++i)
            {
                auto &bbox = bboxes_each_image[i];

                auto class_name = this->mClasses[bbox.cls];
                
                bbox.x1 = bbox.x1 * this->image_cols[b];
                bbox.y1 = bbox.y1 * this->image_rows[b];
                bbox.x2 = bbox.x2 * this->image_cols[b];
                bbox.y2 = bbox.y2 * this->image_rows[b];

                this->mCocoResult << std::fixed << std::setprecision(2);
                this->mCocoResult << "{\"image_id\":" << img_id << ",\"category_id\":" << this->mClassesMap[class_name] << ",\"bbox\":[";
                this->mCocoResult << bbox.x1 << "," << bbox.y1 << "," << bbox.x2 - bbox.x1 << "," << bbox.y2 - bbox.y1;
                this->mCocoResult << std::fixed << std::setprecision(4);
                this->mCocoResult << "],\"score\":" << bbox.score;
                
                if (i == bboxes_each_image.size() - 1 && this->mImageIdx + b == this->mImageFiles.size() - 1)
                {
                    this->mCocoResult << "}\n";
                }
                else
                {
                    this->mCocoResult << "},\n";
                }
            }
        }

        this->mImageIdx += this->mParams.explicitBatchSize;
    }
    else if (this->mParams.demo)
    {
        for (int b = 0; b < this->mParams.explicitBatchSize; ++b)
        {
            auto &bboxes_each_image = nms_bboxes[b];
            std::cout << "------------ Next Image! --------------" << std::endl;
            std::cout << "Number of detections: " << num_detections[b] << std::endl;

            for(auto &bbox : bboxes_each_image)
            {
                std::cout << "[ " << bbox.x1 << " " << bbox.y1 << " " << bbox.x2 << " " << bbox.y2 << " ] score: " << bbox.score << " class: " << bbox.cls << std::endl;
            }
        }

        // Draw bboxes only for the first image in each batch
        cv::Mat bgr_img_cpy = this->mSampleImage.clone();
        this->draw_bboxes(nms_bboxes[0], bgr_img_cpy);
    }

    bool pass = true;

    return pass;
}

std::vector<std::vector<BoundingBox>> SampleYolo::get_bboxes(int batch_size, int keep_topk,
    int32_t *num_detections, float *nmsed_boxes, float *nmsed_scores, float *nmsed_classes)
{
    int n_detect_pos = 0;
    int box_pos = 0;
    int score_pos = 0;
    int cls_pos = 0;

    std::vector<std::vector<BoundingBox>> bboxes {static_cast<size_t>(batch_size)};

    for (int b = 0; b < batch_size; ++b)
    {
        for (int t = 0; t < keep_topk; ++t)
        {
            if (static_cast<int>(nmsed_classes[cls_pos + t]) < 0)
            {
                break;
            }

            int box_coord_pos = box_pos + 4 * t;
            float x1 = nmsed_boxes[box_coord_pos];
            float y1 = nmsed_boxes[box_coord_pos + 1];
            float x2 = nmsed_boxes[box_coord_pos + 2];
            float y2 = nmsed_boxes[box_coord_pos + 3];

            bboxes[b].push_back(BoundingBox {
                std::min(x1, x2),
                std::min(y1, y2),
                std::max(x1, x2),
                std::max(y1, y2),
                nmsed_scores[score_pos + t],
                static_cast<int>(nmsed_classes[cls_pos + t]) });
        }

        n_detect_pos += 1;
        box_pos += 4 * keep_topk;
        score_pos += keep_topk;
        cls_pos += keep_topk;
    }

    return bboxes;
}

void SampleYolo::draw_bboxes(const std::vector<BoundingBox> &bboxes, cv::Mat &testImg)
{
    std::cout << "Writing detection to image ..." << std::endl;
    int H = testImg.rows;
    int W = testImg.cols;

    for(size_t k = 0; k < bboxes.size(); k++)
    {
        if (bboxes[k].cls == -1)
        {
            break;
        }

        int x1 = bboxes[k].x1 * W;
        int y1 = bboxes[k].y1 * H;
        int x2 = bboxes[k].x2 * W;
        int y2 = bboxes[k].y2 * H;

        cv::rectangle(testImg, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 1);

        cv::putText(testImg, //target image
            this->mClasses[bboxes[k].cls], //text
            cv::Point(x1, y1), //top-left position
            cv::FONT_HERSHEY_DUPLEX,
            0.8,
            CV_RGB(118, 185, 0), //font color
            1);
    }

    cv::imwrite(this->mParams.outputImageName, testImg);
}

void SampleYolo::draw_coco_test_bboxes(const std::vector<BoundingBox> &bboxes, cv::Mat &testImg, int img_id)
{
    std::cout << "Writing detection to image ..." << std::endl;

    for(size_t k = 0; k < bboxes.size(); k++)
    {
        if (bboxes[k].cls == -1)
        {
            break;
        }

        int x1 = bboxes[k].x1;
        int y1 = bboxes[k].y1;
        int x2 = bboxes[k].x2;
        int y2 = bboxes[k].y2;

        cv::rectangle(testImg, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 1);

        cv::putText(testImg, //target image
            this->mClasses[bboxes[k].cls], //text
            cv::Point(x1, y1), //top-left position
            cv::FONT_HERSHEY_DUPLEX,
            0.8,
            CV_RGB(118, 185, 0), //font color
            1);
    }

    cv::imwrite(this->mParams.cocoImageOutputDir + std::string("/") + std::to_string(img_id) + std::string("_result.jpg"), testImg);
}

