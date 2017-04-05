//
// Created by Matúš Cimerman on 19/03/2017.
//

#ifndef NUCLEOIDS_DRAW_H
#define NUCLEOIDS_DRAW_H

#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"

namespace Draw {

    cv::Mat drawAndFilterContours(cv::Mat imgOriginal, cv::Mat imgThreshold, int thresholdValue, int filterType = 0,
                               double filterMultiplier = 1);
}

#endif //NUCLEOIDS_DRAW_H