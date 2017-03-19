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

    void drawAndFilterContours(cv::Mat imgOriginal, cv::Mat imgThreshold, int thresholdValue, int filterType,
                               double filterMultiplier) {
        Mat canny_output;
        Mat imgOriginalContours;
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        imgOriginal.copyTo(imgOriginalContours);
        Canny(imgThreshold, canny_output, thresholdValue, thresholdValue * 2, 3);
        findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

        contours = applyContourFilter(contours, filterType, filterMultiplier);

        for (int i = 0; i < contours.size(); i++) {
            drawContours(imgOriginalContours, contours, i, (255, 255, 255), 2, 8, hierarchy, 0, Point());
        }

        namedWindow("Contours", CV_WINDOW_AUTOSIZE);
        imshow("Contours", imgOriginalContours);
    }
}