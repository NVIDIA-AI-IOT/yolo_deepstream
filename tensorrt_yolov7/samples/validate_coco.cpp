
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
#include <json/json.h>
#include <fstream>
#include <argsParser.h>

std::string parse_model_path(argsParser& cmdLine) {
    const char* engine_path_str = cmdLine.ParseString("engine");
    std::string engine_path;
    if (engine_path_str) engine_path = std::string(engine_path_str);
    return engine_path;
}

std::string parse_coco_path(argsParser& cmdLine) {
    const char* coco_path_str = cmdLine.ParseString("coco");
    std::string coco_path;
    if (coco_path_str) coco_path = std::string(coco_path_str);
    return coco_path;
}

bool print_help() {
    printf("--------------------------------------------------------------------------------------------------------\n");
    printf("---------------------------- yolov7 coco validate tool ---------------------------------------------\n");
    printf(" '--help': print help information \n");
    printf(" '--engine=yolov7.engine' Load yolov7 trt-engine  \n");
    printf(" '--coco=./data/coco/' specify the path of the coco dataset\n");
    return true;
}

int coco80_to_coco91_class(int id) {
    //# converts 80-index (val2014) to 91-index (paper)
    // # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    // # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    // # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    // # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    // # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    std::vector<int> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};
    return x[id];
}
std::vector<float> xyxy2xywh(float x0, float x1, float x2, float x3){
    // # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    // y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    std::vector<float> y;
    y.resize(4);
    y[0] = (x0 + x2) / 2;//  # x center
    y[1] = (x1 + x3) / 2;//  # y center
    y[2] = x2 - x0;//  # width
    y[3] = x3 - x1;//  # height
    y[0] -= y[2]/2;
    y[1] -= y[3]/2;
    // box[:, :2] -= box[:, 2:] / 2 
    
    return y;
}

int number_classes = 80;

std::vector<std::string> readCocoPaths(std::string coco_file_path) {
    std::vector<std::string> result;
    std::ifstream coco_test_file(coco_file_path);
    std::string line;
    std::string folder_path = coco_file_path.substr(0, coco_file_path.find_last_of("/")+1);
    if(coco_test_file) {
        while(getline(coco_test_file, line)){
            
            result.push_back(folder_path+line);
            // std::cout<<"folder_path+line:"<<folder_path+line<<std::endl;
        }
    }
    // std::cout <<"Done"<<folder_path<<std::endl;
    return result;
}


int main(int argc, char** argv){

    argsParser cmdLine(argc, argv);
    //! parse device_flag, see parse_device_flag
    if(cmdLine.ParseFlag("help")) { print_help(); return 0; }

    std::string engine_path = parse_model_path(cmdLine);
    std::string coco_path = parse_coco_path(cmdLine);

    coco_path += "/val2017.txt";
    Yolov7 yolov7(engine_path);

    // containor fr
    std::vector<cv::Mat> bgr_imgs;
    std::vector<std::string> imgPathList = readCocoPaths(coco_path);;
    std::vector<std::vector<std::vector<float>>> batchNmsResult;
    int maxBatchsize = yolov7.getInputDim().d[0];

    
    Json::Value root;
    Json::FastWriter writer;

    for(int i = 0 ; i < imgPathList.size(); ){
        //infer with a batch
        for(int j = 0; j < maxBatchsize && i<imgPathList.size() ; j++,i++){
            cv::Mat one_img = cv::imread(imgPathList[i]);
            bgr_imgs.push_back(one_img);
        }

        std::vector<cv::Mat> nchwMats = yolov7.preProcess4Validate(bgr_imgs);

        printf("\r%d / %d", i, imgPathList.size());
        fflush(stdout);
        
        yolov7.infer();
        
        batchNmsResult = yolov7.PostProcess(0.65, 0.001);

        for(int j = 0; j< batchNmsResult.size();j++){
            int imgth = i - batchNmsResult.size() + j;
            // processing the name. eg: ./images/train2017/000000000250.jpg will be processed as 250
            int image_id = stoi(imgPathList[imgth].substr(imgPathList[imgth].length()-16, imgPathList[imgth].find_last_of(".")-(imgPathList[imgth].length()-16)));
            for(int k = 0; k <batchNmsResult[j].size();k++){
                Json::Value OneResult;
                Json::Value bboxObj;
                
                OneResult["image_id"] = image_id;
                OneResult["category_id"] = coco80_to_coco91_class(batchNmsResult[j][k][4]);
                OneResult["score"] = batchNmsResult[j][k][5];
                std::vector<float> point = xyxy2xywh(batchNmsResult[j][k][0],batchNmsResult[j][k][1],batchNmsResult[j][k][2],batchNmsResult[j][k][3]);
                bboxObj.append(point[0]);
                bboxObj.append(point[1]);
                bboxObj.append(point[2]);
                bboxObj.append(point[3]);
                OneResult["bbox"] = bboxObj;
                root.append(OneResult);
            }
        }
        bgr_imgs.clear();
    }
    
    std::string json_file = writer.write(root);
    std::ofstream out("./predict.json");
    out << json_file;
    std::cout<<std::endl<<"predict result has been written to ./predict.json "<<std::endl;
    return 0;
}
