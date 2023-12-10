#include <NvInfer.h>
#include "NvInferPlugin.h"
#include <iostream>
#include<vector>
#include <fstream>
#include <cassert>
#include <string>
#include "FastHandDet.h"
//#include "Nano_track.h"

int main() {
    //std::string model = "../engine/FastestDet_fp16.engine";
    std::string model = "../engine/FastDet_fp32.engine";
    auto hand_dector = new FastHandDet(model, 352, 352, 0.5, 0.1);

    //手部检测模型 需搭配追踪模型一起使用

    cv::VideoCapture cap;
    cv::Mat frame;
    //cap.open("track_text_vidio.mp4");
    cap.open(0);
    int count = 0;
    std::vector<TargetBox> hand_info;
    //如果
    for (;;) {
        count +=1;
        // Read a new frame.
        cap >> frame;
        if (frame.empty())
            break;
        //检测速度较快,其实并不需要每一帧都进行检测    
//        if(count == 3){
        hand_info.clear();
        auto start = std::chrono::high_resolution_clock::now();
        hand_dector->detect(frame, hand_info);
        // 获取结束时间点
        auto end = std::chrono::high_resolution_clock::now();
        // 计算时间差
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // 输出运行时间（毫秒）
        std::cout << "fps: " << 1000/duration.count() << std::endl;
        count =0;
//        }
        for (int i = 0; i < hand_info.size(); i++) {
            //std::cout<<"1"<<std::endl;
            auto hand = hand_info[i];
            //UltraFace的框会大一点,这里将宽度进行了一定程度的缩放.
            cv::Point pt1(hand.x1, hand.y1);
            cv::Point pt2(hand.x2, hand.y2);
            cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
        }
        cv::imshow("hand", frame);
        cv::waitKey(4);
    }
    //手部检测结合追踪
    cv::destroyWindow("UltraFace");
    cap.release();
    delete hand_dector;
    return 0;
}


///手部跟踪仍然采用检测加追踪的方式


