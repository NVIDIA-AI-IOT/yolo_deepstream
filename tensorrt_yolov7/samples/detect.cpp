
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
#include <Yolov7.h>
#include <vector>
#include <numeric>
#include <random>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <argsParser.h>


std::vector<std::string> parse_img_paths(argsParser& cmdLine) {
    return cmdLine.ParseStringList("img");
}

std::string parse_model_path(argsParser& cmdLine) {
    const char* engine_path_str = cmdLine.ParseString("engine");
    std::string engine_path;
    if (engine_path_str) engine_path = std::string(engine_path_str);
    return engine_path;
}

bool print_help() {
    printf("--------------------------------------------------------------------------------------------------------\n");
    printf("---------------------------- yolov7 images detector ---------------------------------------------\n");
    printf(" '--help': print help information \n");
    printf(" '--engine=yolov7.engine' Load yolov7 trt-engine  \n");
    printf(" '--img=img1,jpg,img2.jpg,img3.jpg' specify the path of the images, split by `,`\n");
    return true;
}


int main(int argc, char** argv){

    argsParser cmdLine(argc, argv);
    //! parse device_flag, see parse_device_flag
    if(cmdLine.ParseFlag("help")) { print_help(); return 0; }

    std::string engine_path = parse_model_path(cmdLine);
    std::vector<std::string> img_paths = parse_img_paths(cmdLine);
    
    Yolov7 yolov7(engine_path);

    std::vector<cv::Mat> bgr_imgs;
    for(int i = 0; i< img_paths.size();i++){
        bgr_imgs.push_back(cv::imread(img_paths[i]));
    }
    
    std::cout<<"preprocess start"<<std::endl;

    yolov7.preProcess(bgr_imgs);
    
    std::cout<<"inference start"<<std::endl;

    yolov7.infer();

    std::cout<<"postprocessing start"<<std::endl;

    std::vector<std::vector<std::vector<float>>> nmsresults = yolov7.PostProcess();

    for(int j =0; j < nmsresults.size();j++){
        Yolov7::DrawBoxesonGraph(bgr_imgs[j],nmsresults[j]);
        std::string output_path = img_paths[j] + "detect" + std::to_string(j)+".jpg";      
        cv::imwrite(output_path, bgr_imgs[j]);
        std::cout<<"detectec image written to: "<<output_path<<std::endl;
    }
    
    std::cout<<"Done..."<<std::endl;
    
}
