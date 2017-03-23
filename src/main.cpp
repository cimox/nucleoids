#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/cv.hpp>
#include "core/draw.h"
#include "common/utils.h"

using namespace cv;
using namespace std;

Mat imgOriginal, imgThreshold, imgEroded;

string filename = "../../data/17_cut.png";
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

    // Blob detection
    SimpleBlobDetector::Params params;
    params.minDistBetweenBlobs = 25.0f;
    params.filterByInertia = true;
    params.filterByConvexity = false;
    params.filterByColor = false;
    params.filterByCircularity = false;
    params.filterByArea = true;
    params.minArea = 250.0f;
//    params.maxArea = 1500.0f;
    Ptr<SimpleBlobDetector> d = SimpleBlobDetector::create(params);
    vector<KeyPoint> keypoints;
    d->detect(imgBlurred, keypoints);

    Mat imgBlob;
    drawKeypoints(imgBlurred, keypoints, imgBlob, Scalar(0, 0, 255), DrawMatchesFlags::DEFAULT);
    imshow("Blob", imgBlob);

//    // Apply threshold and show image.
//    threshold(imgBlurred, imgThreshold, thresholdValue, maxThresholdValue, THRESH_BINARY + THRESH_OTSU);
////    adaptiveThreshold(imgBlurred, imgThreshold, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 13, 0.5);
//    imshow(thresholdWindow, imgThreshold);
//
//    // Erosion + dilatation.
//    Mat element = getStructuringElement(MORPH_ELLIPSE,
//                                        Size(2 * erosion_size + 1, 2 * erosion_size + 1),
//                                        Point(erosion_size, erosion_size));
//    erode(imgThreshold, imgEroded, element);
//    dilate(imgEroded, imgEroded, element);
//    imshow("Erosion & dilatation", imgEroded);

    // power-law transformacia

    // do buduca: vymazat jadra buniek, power-law transformacia na zvyraznenie nukleoidov. detekcia nukleoidov a priradenie k centroidov
    // vyskusat Blob - nastavenie parametrov

    // Find and draw contours of threshold's image.
    Draw::drawAndFilterContours(imgOriginal, imgEroded, thresholdValue, Utils::AVERAGE, filterMultiplier);
}