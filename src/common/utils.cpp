//
// Created by Matúš Cimerman on 19/03/2017.
//
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "utils.h"

using namespace std;
using namespace cv;

namespace Utils {

    void printNucleusPositions(Mat imgThreshold) {
        vector<Point> nucleusPositions;
        findNonZero(imgThreshold, nucleusPositions);

        cout << "Nucleus positions: " << nucleusPositions << endl;
    }

    double calculateVectorMedian(vector<double> values) {
        double median;
        size_t size = values.size();

        sort(values.begin(), values.end());

        if (size % 2 == 0)
            median = (values[size / 2 - 1] + values[size / 2]) / 2;
        else
            median = values[size / 2];

        return median;
    }

    double calculateVectorAverage(vector<double> values) {
        double average = 0;

        for (int i = 0; i < values.size(); i++) {
            average += values[i];
        }

        return average / values.size();
    }

    vector<double> getContourAreas(vector<vector<Point>> contours) {
        vector<double> contourAreas;
        for (int i = 0; i < contours.size(); i++) {
            contourAreas.push_back(contourArea(contours[i]));
        }

        return contourAreas;
    }

    vector<vector<Point>>
    filterContours(vector<vector<Point>> contours, double (*metricFunction)(vector<double>), double multiplier) {
        vector<double> contourAreas = getContourAreas(contours);
        vector<vector<Point>> filteredContours;
        double minAreaSize = metricFunction(contourAreas);

        cout << "Metric contour area size " << minAreaSize << endl;

        if (contours.size() <= 1) {
            return contours;
        }

        for (long i = contours.size() - 1; i >= 0; i--) {
            if (contourArea(contours[i]) >= minAreaSize * multiplier) {
                filteredContours.push_back(contours[i]);
            }
        }

        return filteredContours;
    }

    vector<vector<Point>> applyContourFilter(vector<vector<Point>> contours, int filterType, double filterMultiplier) {
        cout << "Filtering " << contours.size() << " contours" << endl;
        switch (filterType) {
            case AVERAGE:
                contours = filterContours(contours, calculateVectorAverage, filterMultiplier);
                break;
            case MEDIAN:
                contours = filterContours(contours, calculateVectorMedian, filterMultiplier);
                break;
            default:
                cout << "Invalid contour filter type" << endl;
                break;
        }
        cout << "Contours " << contours.size() << " left" << endl;

        return contours;
    }
}