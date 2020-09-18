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

#include "SampleYolo.hpp"

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--demo          This app will run demo if this option is set"
              << std::endl;
    std::cout << "--speed         This app will run speed test if this option is set"
              << std::endl;
    std::cout << "--coco          This app will run COCO dataset if this option is set"
              << std::endl;
    std::cout << "--fp16          Specify to run in fp16 mode." << std::endl;
    std::cout << "--int8          Specify to run in int8 mode." << std::endl;
}

SampleYoloParams specifyInputAndOutputNamesAndShapes(SampleYoloParams &params)
{
    params.inputShape = std::vector<int> {params.explicitBatchSize, 3, 416, 416};

    // Output shapes when BatchedNMSPlugin is available
    params.outputShapes.push_back(std::vector<int>{params.explicitBatchSize, 1});
    params.outputShapes.push_back(std::vector<int>{params.explicitBatchSize, params.keepTopK, 4});
    params.outputShapes.push_back(std::vector<int>{params.explicitBatchSize, params.keepTopK});
    params.outputShapes.push_back(std::vector<int>{params.explicitBatchSize, params.keepTopK});

    // Output tensors when BatchedNMSPlugin is available
    params.outputTensorNames.push_back("num_detections");
    params.outputTensorNames.push_back("nmsed_boxes");
    params.outputTensorNames.push_back("nmsed_scores");
    params.outputTensorNames.push_back("nmsed_classes");

    return params;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
SampleYoloParams initializeSampleParams(std::vector<std::string> args)
{
    SampleYoloParams params;

    // This argument is for calibration of int8
    // Int8 calibration is not available until now
    // You have to prepare samples for int8 calibration by yourself 
    params.nbCalBatches = 80;

    // The engine file to generate or to load
    // The engine file does not exist:
    //     This program will try to load onnx file and convert onnx into engine
    // The engine file exists:
    //     This program will load the engine file directly
    params.engingFileName = "../data/yolov4.engine";

    // The onnx file to load
    params.onnxFileName = "../data/yolov4.onnx";
    
    // Input tensor name of ONNX file & engine file
    params.inputTensorNames.push_back("input");
    
    // Old batch configuration, it is zero if explicitBatch flag is true for the tensorrt engine
    // May be deprecated in the future
    params.batchSize = 0;
    
    // Number of classes (usually 80, but can be other values)
    params.outputClsSize = 80;
    
    // topK parameter of BatchedNMSPlugin
    params.topK = 2000;
    
    // keepTopK parameter of BatchedNMSPlugin
    params.keepTopK = 1000;

    // Batch size, you can modify to other batch size values if needed
    params.explicitBatchSize = 1;

    params.inputImageName = "../data/demo.jpg";
    params.cocoClassNamesFileName = "../data/names.txt";
    params.cocoClassIDFileName = "../data/categories.txt";

    // Config number of DLA cores, -1 if there is no DLA core
    params.dlaCore = -1;

    for (auto &arg : args)
    {
        if (arg == "--help")
        {
            printHelpInfo();
        }
        else if (arg == "--demo")
        {
            // Configurations to run a demo image
            params.demo = 1;
            params.outputImageName = "../data/demo_out.jpg";
        }
        else if (arg == "--speed")
        {
            // Configurations to run speed test
            params.speedTest = 1;
            params.speedTestItrs = 1000;
        }
        else if (arg == "--coco")
        {
            // Configurations of Test on COCO dataset
            params.cocoTest = 1;
            params.cocoImageListFileName = "../data/val2017.txt";
            params.cocoTestResultFileName = "../data/coco_result.json";
            params.cocoImageDir = "../data/val2017";
        }
        else if (arg == "--int8")
        {
            params.int8 = true;
        }
        else if (arg == "--fp16")
        {
            params.fp16 = true;
        }
    }

    specifyInputAndOutputNamesAndShapes(params);

    return params;
}

int main(int argc, char** argv)
{
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i)
    {
        args.push_back(std::string(argv[i]));
    }

    auto sampleTest = sample::gLogger.defineTest(SampleYolo::gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    SampleYolo sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for Yolo" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    sample::gLogInfo << "Loading or building yolo model done" << std::endl;

    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    sample::gLogInfo << "Inference of yolo model done" << std::endl;

    if (!sample.teardown())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return EXIT_SUCCESS; // sample::gLogger.reportPass(sampleTest);
}
