//
// Created by Matúš Cimerman on 19/03/2017.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/cv.hpp>
#include "../common/utils.h"

using namespace std;
using namespace cv;
using namespace Utils;

namespace Draw {

    cv::Mat drawAndFilterContours(cv::Mat imgOriginal, cv::Mat imgThreshold, int thresholdValue, int filterType,
                               double filterMultiplier) {
        Mat canny_output;
        Mat imgOriginalContours;
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        imgOriginal.copyTo(imgOriginalContours);
        Canny(imgThreshold, canny_output, thresholdValue, thresholdValue * 2, 3);
        findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

        contours = applyContourFilter(contours, filterType, filterMultiplier);

        int cx, cy;
        for (int i = 0; i < contours.size(); i++) {
            drawContours(imgOriginalContours, contours, i, Scalar(0, 0, 0), 2, LINE_8, hierarchy, 0, Point());
        }

        namedWindow("Contours", CV_WINDOW_AUTOSIZE);
        imshow("Contours", imgOriginalContours);
        moveWindow("Contours", imgOriginal.cols, 1);

        return imgOriginalContours;
    }
}