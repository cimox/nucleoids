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

        int cx, cy;
        for (int i = 0; i < contours.size(); i++) {
            drawContours(imgOriginalContours, contours, i, Scalar(255, 0, 0), 1, LINE_8, hierarchy, 0, Point());

            // Get the moments.
            vector<Moments> mu(contours.size());
            for (int k = 0; k < contours.size(); k++) { mu[k] = moments(contours[k], false); }

            // Get the mass centers.
            vector<Point2f> mc(contours.size());
            for (int k = 0; k < contours.size(); k++) { mc[k] = Point2d(mu[k].m10 / mu[k].m00, mu[k].m01 / mu[k].m00); }

            circle(imgOriginalContours, mc[i], 1, Scalar(0, 0, 255), 8, 0);
        }

        namedWindow("Contours", CV_WINDOW_AUTOSIZE);
        imshow("Contours", imgOriginalContours);
        moveWindow("Contours", imgOriginal.cols, 1);
    }
}