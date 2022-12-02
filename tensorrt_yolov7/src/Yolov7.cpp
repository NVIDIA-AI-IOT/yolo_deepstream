
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


#include "Yolov7.h"

static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

Yolov7::Yolov7(std::string engine_path) {

    this->mTotal_inference_time = 0;
    this->mInference_count = 0;

    this->mStream = makeCudaStream(cudaEventDefault,0);
    this->mEvent = makeCudaEvent(cudaEventDefault);

    this->mEnginePath = engine_path;
    Logger mLoggern;
    initLibNvInferPlugins(&mLoggern, "");
    this->mRuntime = std::unique_ptr<nvinfer1::IRuntime,TrtDeleter<nvinfer1::IRuntime>>{nvinfer1::createInferRuntime(mLoggern)};
    // this->mCudaGraphEnabled = enableCudaGraph;

    std::ifstream fin(engine_path, std::ios::binary);
    std::vector<char> inBuffer((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    fin.close();
    mEngine.reset(mRuntime->deserializeCudaEngine(inBuffer.data(), inBuffer.size(), nullptr));
    mContext.reset(mEngine->createExecutionContext());
    mImgPushed = 0;
    /*
       malloc cuda memory for binding
    */
    const int nbBindings = this->mEngine->getNbBindings();

    this->mDynamicBatch =  this->mEngine->getBindingDimensions(0).d[0] == -1 ? true:false;
    for (int i = 0; i < nbBindings; i++) {
        const auto dataType = this->mEngine->getBindingDataType(i);
        const int elemSize = [&]() -> int {
            switch (dataType) {
            case nvinfer1::DataType::kFLOAT:
                return 4;
            case nvinfer1::DataType::kHALF:
                return 2;
            default:
                throw std::runtime_error("invalid data type");
            }
        }();

        nvinfer1::Dims dims;

        //input
        if (mEngine->bindingIsInput(i)) {
            if(this->mDynamicBatch) 
                dims = mEngine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            else
                dims = mEngine->getBindingDimensions(i);
            this->mInputDim = dims;
        }
        else{ // output
            dims = mEngine->getBindingDimensions(i);
            //if dynamic batch, change dim[0] to max-batchsize of input
            if(this->mDynamicBatch)
                dims.d[0] = mEngine->getProfileDimensions(0, 0, nvinfer1::OptProfileSelector::kMAX).d[0];

            this->mOutputDim = dims;
        }
        const int bindingSize = elemSize * std::accumulate(dims.d, &dims.d[dims.nbDims], 1, std::multiplies<int>{});
        if (mEngine->bindingIsInput(i)) //intput
            mImgBufferSize = bindingSize / dims.d[0];
        else //output
            mHostOutputBuffer.resize(bindingSize / elemSize);

        this->mBindings.emplace_back(mallocCudaMem<char>(bindingSize));
        this->mBindingArray.emplace_back(mBindings.back().get());
    }
    mMaxBatchSize = mInputDim.d[0];

    this->ReportArgs();
    
    if(this->mDynamicBatch)
        mContext->setOptimizationProfileAsync(0, mStream.get());

    return;
}

void Yolov7::ReportArgs() {
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Yolov7 initialized from: " << mEnginePath << std::endl;
    const int nbBindings = mEngine->getNbBindings();
    for (int i = 0; i < nbBindings; i++) {
        const auto dims = mEngine->getBindingDimensions(i);
        if (mEngine->bindingIsInput(i))
            std::cout << "input : " << mEngine->getBindingName(i);
        else
            std::cout << "output : " << mEngine->getBindingName(i);
        std::cout << " , shape : [ ";
        for (int j = 0; j < dims.nbDims; j++) std::cout << dims.d[j] << ",";
        std::cout << "]" << std::endl;
    }
    std::cout << "--------------------------------------------------------" << std::endl;
}

static void hwc_to_chw(cv::InputArray src, cv::OutputArray dst) {
  std::vector<cv::Mat> channels;
  cv::split(src, channels);
  // Stretch one-channel images to vector
  for (auto &img : channels) {
    img = img.reshape(1, 1);
  }
  // Concatenate three vectors to one
  cv::hconcat( channels, dst );
}

std::vector<cv::Mat> Yolov7::preProcess(std::vector<cv::Mat> &cv_img) {
    if(cv_img.size() > mInputDim.d[0] || cv_img.size() <=0) {
        std::cerr<<"error cv_img.size() in "<<__FUNCTION__<<std::endl;
    }

    std::vector<cv::Mat> nchwMats;
    
    for(int i = 0; i< cv_img.size();i++){
        float scale_x = mInputDim.d[3] / (float)cv_img[i].cols;
        float scale_y = mInputDim.d[2] / (float)cv_img[i].rows;
        float scale = std::min(scale_x, scale_y);
        float i2d[6], d2i[6];

        // resize the image, the src img and the dst img have the same center
        i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * cv_img[i].cols + mInputDim.d[3] + scale  - 1) * 0.5;
        i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * cv_img[i].rows + mInputDim.d[2] + scale - 1) * 0.5;

        cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);  // image to dst(network), 2x3 matrix
        cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);  // dst to image, 2x3 matrix
        cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        std::vector<float> d2i_1{d2i[0],d2i[1],d2i[2],d2i[3],d2i[4],d2i[5]};
        this->md2i.push_back(d2i_1);
        cv::Mat input_image;
        cv::cvtColor(cv_img[i], input_image, cv::COLOR_BGR2RGB);
        cv::warpAffine(input_image, input_image, m2x3_i2d, cv::Size(mInputDim.d[3], mInputDim.d[2]), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));
        input_image.convertTo(input_image, CV_32FC3, 1.0f/255.0f, 0);
        cv::Mat nchwMat;
        hwc_to_chw(input_image, nchwMat);
        nchwMats.push_back(nchwMat);
        this->pushImg(nchwMat.data, 1 ,true);
    }
    return nchwMats;
}
std::vector<cv::Mat> Yolov7::preProcess4Validate(std::vector<cv::Mat> &cv_img) {
    std::vector<cv::Mat> nchwMats;
    if(cv_img.size() > mInputDim.d[0] || cv_img.size() <=0) {
        std::cerr<<"error cv_img.size() in "<<__FUNCTION__<<std::endl;
        return nchwMats;
    }
    if( !(mInputDim.d[1]==3 && mInputDim.d[2] == 672 && mInputDim.d[3] == 672)){
        std::cerr<<"for validate, image size must = [batchsize, 3, 672, 672]! "<<std::endl;
        return nchwMats;    
    }

    for(int i = 0; i< cv_img.size();i++){
        std::vector<float> from_{cv_img[i].cols,cv_img[i].rows};
        std::vector<float> to_{640, 640};
        float scale = to_[0]/from_[0] < to_[1]/from_[1]? to_[0]/from_[0]:to_[1]/from_[1];
        std::vector<float> M_{scale, 0, -scale * from_[0]  * 0.5  + to_[0] * 0.5 + scale * 0.5 - 0.5 + 16, 
                           0, scale, -scale * from_[1] * 0.5 + to_[1] * 0.5 + scale * 0.5 - 0.5 + 16};
        

        cv::Mat M(2,3,CV_32FC1, M_.data());
        float d2i[6];
        cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);  // dst to image, 2x3 matrix
        cv::invertAffineTransform(M, m2x3_d2i);
        std::vector<float> d2i_1{d2i[0],d2i[1],d2i[2],d2i[3],d2i[4],d2i[5]};
        md2i.push_back(d2i_1);

        cv::Mat input_image;
        cv::Mat nchwMat;

        cv::Scalar scalar = cv::Scalar::all(114);
        cv::cvtColor(cv_img[i], input_image, cv::COLOR_BGR2RGB);
        cv::warpAffine(input_image, input_image, M,cv::Size(672,672),cv::INTER_LINEAR,cv::BORDER_CONSTANT, scalar);

        input_image.convertTo(input_image, CV_32FC3, 1.0f/255.0f, 0);
        
        hwc_to_chw(input_image, nchwMat);
        nchwMats.push_back(nchwMat);
        this->pushImg(nchwMat.data, 1 ,true);
    }
    return nchwMats;
    
}
int Yolov7::pushImg(void *imgBuffer, int numImg, bool fromCPU) {
    if(mImgPushed + numImg > mMaxBatchSize) {
        std::cerr <<" error: mImgPushed = "<< mImgPushed <<" numImg = "<<numImg<<" mMaxBatchSize= "<< mMaxBatchSize<<", mImgPushed + numImg > mMaxBatchSize "<<std::endl;
    }
    if(fromCPU) {
        checkCudaErrors(cudaMemcpy(this->mBindings[0].get() + mImgPushed*mImgBufferSize, imgBuffer, mImgBufferSize * numImg, cudaMemcpyHostToDevice));
    }
    else {
        checkCudaErrors(cudaMemcpy(this->mBindings[0].get() + mImgPushed*mImgBufferSize, imgBuffer, mImgBufferSize * numImg, cudaMemcpyDeviceToDevice));
    }
    mImgPushed += numImg;
    
    return 0;
}

int Yolov7::infer() {
    if(mImgPushed == 0){
        std::cerr <<" error: mImgPushed = "<< mImgPushed <<"  ,mImgPushed == 0!"<<std::endl;
        return -1;
    }
    nvinfer1::Dims inferDims = mInputDim;

    if(mDynamicBatch) {
        inferDims.d[0] = mImgPushed;
        this->mContext->setBindingDimensions(0, inferDims);
    }

    if (!mContext->enqueueV2(mBindingArray.data(), mStream.get(), NULL)) {
        std::cout << "failed to enqueue TensorRT context on device "<< std::endl;
        return -1;
    }

    if (cudaSuccess != cudaStreamSynchronize(mStream.get())) {
        std::cout << "Stream Sync failed "<< std::endl;
        return -1;
    }
    mImgPushed = 0;
    mInference_count++;
    return 0;
}


nvinfer1::Dims Yolov7::getInputDim() {
    return mInputDim;
}

nvinfer1::Dims Yolov7::getOutputDim() {
    return mOutputDim;
}
std::vector<std::vector<std::vector<float>>> Yolov7::decode_yolov7_result(float conf_thres) {
    // for now, copy all buffer to host
    std::vector<std::vector<std::vector<float>>> all_bboxes;
    if(cudaSuccess != cudaMemcpyAsync((void*)(mHostOutputBuffer.data()),this->mBindings[1].get() , sizeof(float) * mHostOutputBuffer.size(), cudaMemcpyDeviceToHost)){
        std::cerr<<"error cv_img.size() in "<<__FUNCTION__<<std::endl;
        return all_bboxes;// blank result
    }
    std::vector<std::vector<float>> bboxes;
    float *h_one_output;
    for(int j = 0; j < md2i.size();j++){
        bboxes.clear();
        h_one_output = mHostOutputBuffer.data() + j * std::accumulate(&mOutputDim.d[1], &mOutputDim.d[mOutputDim.nbDims], 1, std::multiplies<int>{});
        // float conf_thres = 0.4;

        int output_numbox = mOutputDim.d[1];
        int output_numprob = mOutputDim.d[2];
        int num_classes = output_numprob - 5;

        for(int i = 0; i < output_numbox; ++i){
            float* ptr = h_one_output + i * output_numprob;
            float objness = ptr[4];
            if(objness < conf_thres)
                continue;

            float* pclass = ptr + 5;
            int label     = std::max_element(pclass, pclass + num_classes) - pclass;
            float prob    = pclass[label];
            float confidence = prob * objness;
            if(confidence < conf_thres)
                continue;

            // center point, width, height
            float cx     = ptr[0];
            float cy     = ptr[1];
            float width  = ptr[2];
            float height = ptr[3];

            // predict box
            float left   = cx - width * 0.5;
            float top    = cy - height * 0.5;
            float right  = cx + width * 0.5;
            float bottom = cy + height * 0.5;

            // the position on the picture
            float image_base_left   = md2i[j][0] * left   + md2i[j][2];
            float image_base_right  = md2i[j][0] * right  + md2i[j][2];
            float image_base_top    = md2i[j][0] * top    + md2i[j][5];
            float image_base_bottom = md2i[j][0] * bottom + md2i[j][5];
            bboxes.push_back({image_base_left, image_base_top, image_base_right, image_base_bottom, (float)label, confidence});
        }
        all_bboxes.push_back(bboxes);
    }
    md2i.clear();
    return all_bboxes;
}

std::vector<std::vector<float>> Yolov7::nms(std::vector<std::vector<float>> &bboxes, float iou_thres) {
    std::sort(bboxes.begin(), bboxes.end(), [](std::vector<float>& a, std::vector<float>& b){return a[5] > b[5];});
    std::vector<bool> remove_flags(bboxes.size());
    std::vector<std::vector<float>> box_result;
    box_result.reserve(bboxes.size());

    auto iou = [](const std::vector<float>& a, const std::vector<float>& b){
        float cross_left   = std::max(a[0], b[0]);
        float cross_top    = std::max(a[1], b[1]);
        float cross_right  = std::min(a[2], b[2]);
        float cross_bottom = std::min(a[3], b[3]);

        float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
        float union_area = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]) 
                         + std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]) - cross_area;
        if(cross_area == 0 || union_area == 0) return 0.0f;
        return cross_area / union_area;
    };

    for(int i = 0; i < bboxes.size(); ++i){
        if(remove_flags[i]) continue;

        auto& ibox = bboxes[i];
        box_result.emplace_back(ibox);
        for(int j = i + 1; j < bboxes.size(); ++j){
            if(remove_flags[j]) continue;

            auto& jbox = bboxes[j];
            if(ibox[4] == jbox[4]){
                // class matched
                if(iou(ibox, jbox) >= iou_thres)
                    remove_flags[j] = true;
            }
        }
    }
    return box_result;
}
std::vector<std::vector<std::vector<float>>> Yolov7::yolov7_nms(std::vector<std::vector<std::vector<float>>> &bboxes, float iou_thres) {
    std::vector<std::vector<std::vector<float>>> nms_result;
    for(int i = 0;i < bboxes.size();i++) {
        nms_result.push_back(this->nms(bboxes[i], iou_thres));
    }
    return nms_result;
}

std::vector<std::vector<std::vector<float>>> Yolov7::PostProcess(float iou_thres, float conf_thres){
    std::vector<std::vector<std::vector<float>>> PostProcessingResult;
    //decode & nms
    std::vector<std::vector<std::vector<float>>> decode_result = this->decode_yolov7_result(conf_thres);
    PostProcessingResult = this->yolov7_nms(decode_result, iou_thres);    
    return PostProcessingResult;
}

//help functions for drawing boxes on cv::Mat
static std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v){
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f*s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) {
    case 0:r = v; g = t; b = p;break;
    case 1:r = q; g = v; b = p;break;
    case 2:r = p; g = v; b = t;break;
    case 3:r = p; g = q; b = v;break;
    case 4:r = t; g = p; b = v;break;
    case 5:r = v; g = p; b = q;break;
    default:r = 1; g = 1; b = 1;break;}
    return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}
static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id){
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

static int Yolov7::DrawBoxesonGraph(cv::Mat &bgr_img, std::vector<std::vector<float>> nmsresult){
    for(int i = 0; i < nmsresult.size(); ++i){
        auto& ibox = nmsresult[i];
        float left = ibox[0];
        float top = ibox[1];
        float right = ibox[2];
        float bottom = ibox[3];
        int class_label = ibox[4];
        float confidence = ibox[5];
        cv::Scalar color;
        std::tie(color[0], color[1], color[2]) = random_color(class_label);
        cv::rectangle(bgr_img, cv::Point(left, top), cv::Point(right, bottom), color, 3);

        auto name      = cocolabels[class_label];
        auto caption   = cv::format("%s %.2f", name, confidence);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(bgr_img, cv::Point(left-3, top-33), cv::Point(left + text_width, top), color, -1);
        cv::putText(bgr_img, caption, cv::Point(left, top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    return 0;
}
