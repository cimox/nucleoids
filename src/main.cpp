#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/cv.hpp>
#include "core/draw.h"
#include "common/utils.h"

using namespace cv;
using namespace std;

Mat imgOriginal, imgThreshold, imgEroded, imgBlob;

string FILENAME = "../../data/samples/14.tiff";
int MAX_THRESHOLD = 255, THRESHOLD = 0, THRESHOLD_TYPE = 8;
double FILTER_MULTIPLIER = 0;
int EROSION_SIZE = 11;

void findNucleus();
void closingAndContours();

int main() {
    string originalWindow = "Original", thresholdWindow = "Threshold";
    imgOriginal = imread(FILENAME, IMREAD_COLOR); // load image in grayscale

    if (imgOriginal.empty()) {
        cout << "Could not open imgOriginal!" << std::endl;
        return -1;
    }

    if (THRESHOLD_TYPE == 8) {
        THRESHOLD_TYPE = THRESH_BINARY | THRESHOLD_TYPE;
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
    params.minDistBetweenBlobs = 50.0f;
    params.filterByInertia = false;
    params.filterByConvexity = false;
    params.filterByColor = false;
    params.filterByCircularity = false;
    params.filterByArea = true;
    params.minArea = 250.0f;
    Ptr<SimpleBlobDetector> d = SimpleBlobDetector::create(params);
    vector<KeyPoint> keypoints;

    // Apply threshold and show image.
    threshold(imgBlurred, imgThreshold, THRESHOLD, MAX_THRESHOLD, THRESH_BINARY + THRESH_OTSU);

    // Blob detection.
    d->detect(imgOriginal, keypoints);
    drawKeypoints(imgOriginal, keypoints, imgBlob, Scalar(0, 0, 255), DrawMatchesFlags::DEFAULT);
    imshow("Blob", imgBlob);
    moveWindow("Blob", imgOriginal.cols, 5);


    // do buduca: vymazat jadra buniek,                         TODO
    // power-law transformacia na zvyraznenie nukleoidov.       TODO
    // detekcia nukleoidov a priradenie k centroidov            TODO
    // vyskusat Blob - nastavenie parametrov                    DONE
}

void closingAndContours() {
    // Erosion + dilatation.
    Mat element = getStructuringElement(MORPH_ELLIPSE,
                                        Size(2 * EROSION_SIZE + 1, 2 * EROSION_SIZE + 1),
                                        Point(EROSION_SIZE, EROSION_SIZE));
    erode(imgThreshold, imgEroded, element);
    dilate(imgEroded, imgEroded, element);
    imshow("Erosion & dilatation", imgEroded);

    // Find and draw contours of threshold's image.
    Draw::drawAndFilterContours(imgOriginal, imgEroded, THRESHOLD, Utils::AVERAGE, FILTER_MULTIPLIER);
}