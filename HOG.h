//
// Created by gorigan on 8/28/15.
//

#ifndef HOGEXPERIMENT_HOG_H
#define HOGEXPERIMENT_HOG_H
#include <cassert>
#include <string>
#include <vector>
#include <opencv/cv.h>
#include <opencv/highgui.h>

namespace patreo {
    class HOG {
        std::string inpModSamples_;
        std::string structureFile_;
        std::string outDir_;
        void Execute();
        cv::Mat getHogDescriptor(int x, int y, int w, int h,
                                             int ncells_rows, int ncells_cols,
                                             const std::vector<cv::Mat>& integralGradImage);
        void writeDescriptor (const std::vector<cv::Mat>& descriptors, const std::vector<cv::Rect> &rects,
        const std::string &outDir, std::string& outputname);
        std::vector<cv::Mat> getIntegralGradientImage(const cv::Mat &img);
        //given a width and height it generates nSamples
        std::vector<cv::Rect> sampleImage(
                const int width, const int height,
                float winWidth, float winHeight,
                const float minScale, const float maxScale, int nScales = 0,
                float deltaScale = 0, int nSamples = 0,
                float strideX = 1.0f, float strideY = 1.0f);


    };

}

#endif //HOGEXPERIMENT_HOG_H
