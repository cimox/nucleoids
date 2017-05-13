//
// Created by Matúš Cimerman on 05/04/2017.
//

#ifndef NUCLEOIDS_OPERATIONS_H
#define NUCLEOIDS_OPERATIONS_H

#include <opencv2/core/mat.hpp>

namespace Operations {

    void preprocessImage(cv::Mat &imgSrc, cv::Mat &imgDst, bool showImg = false);

    void morphOpening(cv::Mat &imgSrc, cv::Mat &imgDst, int erosion_size, bool showImg = false);

    void gammaCorrection(cv::Mat &imgSrc, cv::Mat &imgDst, float fGamma, bool showImg = false);

    std::vector<cv::KeyPoint>
    simpleBlobDetection(cv::Mat &imgSrc, cv::Mat &imgDst, std::string windowName, bool showImg = false);

    std::vector<cv::KeyPoint>
    countNucleus(cv::Mat &imgSrc, cv::Mat &imgDst, std::string windowName, bool showImg = false);
}


#endif //NUCLEOIDS_OPERATIONS_H
