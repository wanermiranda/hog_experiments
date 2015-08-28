//
// Created by gorigan on 8/28/15.
//

#include "HOG.h"

using namespace cv;

namespace patreo {
    static cv::Mat getFeats (const std::string& input);

    std::vector<cv::Rect> HOG::sampleImage(const int width,
                                           const int height,
                                           float winWidth, float winHeight,
                                           const float minScale, const float maxScale,
                                           int nScales, float deltaScale,
                                           int nSamples, float strideX, float strideY) {
        assert (width > 0);
        assert (height > 0);

        assert(nScales || deltaScale);
        if (deltaScale < FLT_EPSILON) {
            deltaScale = pow((maxScale / minScale), 1 / static_cast<float>(nScales));
        }
        //int samples = ((height / winHeight)*(width / winWidth)) / (80 * 80)*nScales;
        std::vector<cv::Rect> rects;
        for (float scale = minScale; scale < maxScale; scale *= deltaScale) {
            int h = static_cast<int>(winHeight * scale),
                    w = static_cast<int>(winWidth * scale);
            for (int y = 0; y < height - h; y += static_cast<int>(strideY * h)) {
                for (int x = 0; x < width - w; x += static_cast<int>(strideX * w)) {
                    cv::Rect rect(x, y, w, h);
                    rects.push_back(rect);
                }
            }
        }
        assert (rects.size() > 0);
        return rects;
    }

    std::vector<cv::Mat> HOG::getIntegralGradientImage(const cv::Mat &img) {
        cv::HOGDescriptor hogCalculator;
        cv::Mat grad, angleOfs;
        int rows, cols = 0;
        std::vector<cv::Mat> integralImages;
        hogCalculator.computeGradient(img, grad, angleOfs);
        rows = img.rows, cols = img.cols;
        integralImages.resize(9);
        for (int bin = 0; bin < 9; ++bin) {
            integralImages[bin].create(rows, cols, CV_32F);
        }
        std::vector<cv::Mat> angles;
        std::vector<cv::Mat> gradients;
        cv::split(grad, gradients);
        cv::split(angleOfs, angles);
        for (int i = 0; i < grad.rows; ++i) {
            for (int j = 0; j < grad.cols; ++j) {
                for (int k = 0; k < 2; ++k) {
                    auto angle = angles[k];
                    auto bin = (angle.at<uint8_t>(i, j));
                    auto bingrad = gradients[k];
                    float mag = bingrad.at<float>(i, j);
                    integralImages[bin].at<float>(i, j) = mag;
                }
            }
        }
        for (int bin = 0; bin < 9; ++bin) {
            cv::Mat intImage;
            cv::integral(integralImages[bin], intImage);
            integralImages[bin] = intImage;
        }
        return integralImages;
    }

    cv::Mat HOG::getHogDescriptor(int x, int y, int w, int h, int ncells_rows, int ncells_cols,
                                  const std::vector<cv::Mat> &integralGradImage) {
        //TODO: FIX
        float cellWidth = w / static_cast<float>(ncells_cols);
        float cellHeight = h / static_cast<float>(ncells_rows);
        {
            int rows, cols = 0;
            rows = integralGradImage[0].rows, cols = integralGradImage[0].cols;
            assert((h <= rows) && (w <= cols));
        }
        cv::Mat ans;
        ans.create(1, 9 * ncells_cols * ncells_rows, CV_32F);
        for (int j = 0; j < ncells_rows; ++j) {
            for (int i = 0; i < ncells_cols; ++i) {
                for (int bin = 0; bin < 9; ++bin) {
                    int binOffset = 2 * j + i;
                    int row1, col1, row2, col2, row3, col3, row4, col4;
                    row1 = y + j * static_cast<int>(cellWidth);
                    col1 = x + static_cast<int>(cellHeight) * i;
                    row2 = row1;
                    col2 = col1 + static_cast<int>(cellWidth);
                    row3 = row1 + static_cast<int>(cellHeight);
                    col3 = col1;
                    row4 = col1 + static_cast<int>(cellWidth);
                    col4 = row1 + static_cast<int>(cellHeight);

                    auto binIntImg = integralGradImage[bin];
                    ans.at<float>(0,bin + (binOffset * 9)) =
                            binIntImg.at<float>(row1,col1) + binIntImg.at<float>(row4,col4)-
                            (binIntImg.at<float>(row2,col2) + binIntImg.at<float>(row3,col3));
                }
            }
        }
        float l2norm = static_cast<float>(cv::norm(ans, cv::NORM_L2));
        ans = ans / (l2norm + FLT_EPSILON);
        cv::checkRange(ans, false);
        return ans;
    }

    void HOG::writeDescriptor(
            const std::vector<cv::Mat> &descriptors,
            const std::vector<cv::Rect> &rects,
            const std::string &outDir, std::string &outputname) {

        std::string imgName = outputname.substr(outputname.find_last_of("\\") + 1);
        imgName = imgName.substr(0, imgName.find_last_of("."));
        const string outName;// = outDir + imgName + ".yml";

        cv::FileStorage stg;
        stg.open(outName, cv::FileStorage::WRITE);
        stg << "mats" << "[";
        for (size_t i = 0; i < descriptors.size(); ++i) {
            int pos = static_cast<int>(i);
            cv::Mat m = descriptors[pos];
            stg << m;
        }
        stg << "]";
        stg.release();

        auto feats = getFeats(outName);

        for (int i = 0; i < static_cast<int>(descriptors.size()); ++i) {
            for (int col = 0; col < descriptors[i].cols; ++col) {
                float a = feats.at<float>(i, col);
                auto b = descriptors[i].at<float>(0,col);
                assert (a == b);
            }
        }
    }

    void HOG::Execute() {
        cv::Mat gradImg, angleOfs;
        cv::FileStorage datasetStructure;
        datasetStructure.open(structureFile_, FileStorage::READ);
        auto mainNode = datasetStructure["files"];
        int nImages = 0, nPatches = 0;
        for (auto it = mainNode.begin(); it != mainNode.end(); ++it) {
            assert(!(*it).isNone());

            std::string filename = std::string((*it));
            cv::Mat imgMat = cv::imread(filename);
            printf("Processing [%s]\n", filename.c_str());
            auto patches = sampleImage(imgMat.cols,
                                       imgMat.rows, 80, 80, 1.0f, 2.0f, 7, 0.0f, 50000, 0.4f, 0.4f);
            auto intImage = getIntegralGradientImage(imgMat);
            std::vector<cv::Mat> patchesDescriptors;
            patchesDescriptors.clear();
            for (auto patch : patches) {
                int patchW = patch.width, patchH = patch.height;
                assert (patchW > 0);
                assert (patchH > 0);
                auto hogDescriptor = getHogDescriptor(patch.x, patch.y, patchW, patchH, 2, 2, intImage);
                assert (hogDescriptor.cols == 36);
                patchesDescriptors.push_back(hogDescriptor);
            }
            if (patchesDescriptors.empty())continue;
            writeDescriptor(patchesDescriptors, patches, outDir_, filename);
            patchesDescriptors.clear();
            nPatches += static_cast<int>(patches.size());
            patches.clear();
            ++nImages;
        }
        float avgPatches = nPatches / static_cast<float>(nImages);
        printf("nImages proccessed[%d] avg Patches per Image[%f]\n", nImages, avgPatches);
    }

    cv::Mat getFeats (const std::string& input) {
    int dimensions = 36;
    cv::Mat featMat;

    cv::FileStorage in;
    in.open (input, cv::FileStorage::READ);

    auto matsNode = in["mats"];

    for(auto it = matsNode.begin (); it != matsNode.end (); ++it) {
        cv::Mat mat;
    (*it) >> mat;
    featMat.push_back (mat);
}

return featMat;
}
}