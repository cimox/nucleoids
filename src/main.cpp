#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

Mat imgOriginal, imgThreshold;
string originalWindow = "Original", thresholdWindow = "Threshold";
string trackbarThresholdValue = "Threshold value", trackbarThresholdType = "Threshold type";
int maxThresholdValue = 255, thresholdValue = 0, maxThresholdType = 4, thresholdType = 3;

void thresholdCallback(int, void*);

int main() {
    imgOriginal = imread("../../data/5-6/005-Z_cut.tif", IMREAD_GRAYSCALE); // load image in grayscale

    if (imgOriginal.empty()) {
        cout << "Could not open imgOriginal!" << std::endl;
        return -1;
    }

    // Window with original image.
    namedWindow(originalWindow, WINDOW_AUTOSIZE);
    imshow(originalWindow, imgOriginal);

    // Window with image after threshold.
    namedWindow(thresholdWindow, WINDOW_AUTOSIZE);
    createTrackbar(trackbarThresholdValue, originalWindow, &thresholdValue, maxThresholdValue, thresholdCallback);
    createTrackbar(trackbarThresholdType, originalWindow, &thresholdType, maxThresholdType, thresholdCallback);

    waitKey(0);
    return 0;
}

void thresholdCallback(int, void*)
{
    cout << "threshold value: " << thresholdValue << endl;
    threshold(imgOriginal, imgThreshold, thresholdValue, 255, thresholdType);
    imshow(thresholdWindow, imgThreshold);
}

