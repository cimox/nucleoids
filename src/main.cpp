#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "core/draw.h"
#include "common/utils.h"

using namespace cv;
using namespace std;

Mat imgOriginal, imgThreshold, imgEroded;

string filename = "../../data/samples/9.tiff";
string originalWindow = "Original", thresholdWindow = "Threshold";
int maxThresholdValue = 255,
        thresholdValue = 0,
        thresholdType = 8;
double filterMultiplier = 0;
int erosion_size = 11;

void findNucleus();


int main() {
    imgOriginal = imread(filename, IMREAD_COLOR); // load image in grayscale

    if (imgOriginal.empty()) {
        cout << "Could not open imgOriginal!" << std::endl;
        return -1;
    }

    if (thresholdType == 8) {
        thresholdType = THRESH_BINARY | thresholdType;
    }

    // Window with original image.
    namedWindow(originalWindow, WINDOW_AUTOSIZE);
    imshow(originalWindow, imgOriginal);

    // Window with image after threshold.
    namedWindow(thresholdWindow, WINDOW_AUTOSIZE);
    findNucleus();

    waitKey(0);
    return 0;
}

void findNucleus() {
    Mat imgOriginalGrey, imgBlurred;

    // Convert to greyscale.
    cvtColor(imgOriginal, imgOriginalGrey, COLOR_BGR2GRAY);

    // Apply gaussian blur.
    GaussianBlur(imgOriginalGrey, imgBlurred, Size(11, 11), 0);

    // Apply threshold and show image.
    threshold(imgBlurred, imgThreshold, thresholdValue, maxThresholdValue, THRESH_BINARY + THRESH_OTSU);
//    adaptiveThreshold(imgBlurred, imgThreshold, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 7, 1);
    imshow(thresholdWindow, imgThreshold);
    moveWindow(thresholdWindow, imgOriginal.cols, 1);

    // Erosion + dilatation.
    Mat element = getStructuringElement(MORPH_ELLIPSE,
                                        Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                        Point(erosion_size, erosion_size));
    erode(imgThreshold, imgEroded, element);
    dilate(imgEroded, imgEroded, element);
    imshow("Erosion & dilatation", imgEroded);

    // Find and draw contours of threshold's image.
    Draw::drawAndFilterContours(imgOriginal, imgEroded, thresholdValue, Utils::AVERAGE, filterMultiplier);
}