//
// Created by Matúš Cimerman on 19/03/2017.
//

#ifndef NUCLEOIDS_UTILS_H
#define NUCLEOIDS_UTILS_H

#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"

namespace Utils {

    enum FilterType {
        AVERAGE = 0,
        MEDIAN = 1
    };

    void printNucleusPositions(cv::Mat imgThreshold);

    double calculateVectorMedian(std::vector<double> values);

    double calculateVectorAverage(std::vector<double> values);

    std::vector<double> getContourAreas(std::vector<std::vector<cv::Point>> contours);

    std::vector<std::vector<cv::Point>>
    filterContours(std::vector<std::vector<cv::Point>> contours, double (*metricFunction)(std::vector<double>),
                   double multiplier);

    std::vector<std::vector<cv::Point>>
    applyContourFilter(std::vector<std::vector<cv::Point> > contours, int filterType, double filterMultiplier);


}


#endif //NUCLEOIDS_UTILS_H