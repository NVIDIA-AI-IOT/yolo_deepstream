
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


std::string parse_video_path(argsParser& cmdLine) {
    const char* video_path_str = cmdLine.ParseString("video");
    std::string video_path;
    if (video_path_str) video_path = std::string(video_path_str);
    return video_path;
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
    printf(" '--video=video.mp4' specify the path of the video \n");
    return true;
}


int main(int argc, char** argv){

    argsParser cmdLine(argc, argv);
    //! parse device_flag, see parse_device_flag
    if(cmdLine.ParseFlag("help")) { print_help(); return 0; }

    std::string engine_path = parse_model_path(cmdLine);
    std::string video_path = parse_video_path(cmdLine);
    
    Yolov7 yolov7(engine_path);

    cv::VideoCapture capture;
    cv::Mat frame;
    frame= capture.open(video_path);
    if(!capture.isOpened())
    {
        printf("can not open ... please check whether your opencv has installed with ffmpeg..\n");
        return -1;
    }
    cv::Size size = cv::Size(capture.get(cv::CAP_PROP_FRAME_WIDTH), capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter writer;
    writer.open(std::string(video_path+".detect.mp4"), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, size, true);
    std::vector<cv::Mat> framev;
    std::vector<std::vector<std::vector<float>>> nmsresults;
    int total_frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
    int i = 0;
    while (capture.read(frame)){
        framev.push_back(frame);
        yolov7.preProcess(framev);
        yolov7.infer();
        nmsresults = yolov7.PostProcess();
        Yolov7::DrawBoxesonGraph(frame,nmsresults[0]);
        writer.write(frame);
        framev.clear();
        i++;
        printf("\r%d / %d", i, total_frame_count);
        fflush(stdout);
    }
    capture.release();
    std::cout<<"Done..."<<std::endl;
    
}
