//
// Created by Matúš Cimerman on 05/04/2017.
//

#ifndef NUCLEOIDS_OPERATIONS_H
#define NUCLEOIDS_OPERATIONS_H

#include <opencv2/core/mat.hpp>

namespace Operations {

    void preprocessImage(cv::Mat &imgSrc, cv::Mat &imgDst);

    void morphClosing(cv::Mat &imgSrc, cv::Mat &imgDst, int erosion_size, bool showImg);

    void gammaCorrection(cv::Mat &imgSrc, cv::Mat &imgDst, float fGamma);

    std::vector<cv::KeyPoint> simpleBlobDetection(cv::Mat &imgSrc, cv::Mat &imgDst, bool showImg = false);
}


#endif //NUCLEOIDS_OPERATIONS_H
