#include "iostream"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv/cv.hpp"
#include "core/draw.h"
#include "common/utils.h"
#include "core/operations.h"

using namespace cv;
using namespace std;

Mat imgOriginal, imgPreprocessed, imgPlaceholder, imgSubtracted, imgBinarized;
string FILENAME = "../../data/samples/31.tiff";
int GAMMA_VALUE = 10;
int thresholdValue = 0, thresholdBlockSize = 5, thresholdC = 0;
int medianSize = 5, elementSize = 3;
bool blurGammaMask = false, showAllImages = false;

void removeNucleus(cv::Mat &imgSrc, cv::Mat &imgDst, bool showImg = false);

void updateThreshold(int, void *);

void blob(cv::Mat &imgSrc, cv::Mat &imgDst) {
    cv::SimpleBlobDetector::Params params;
//    params.minDistBetweenBlobs = 50.0f;

    params.filterByInertia = true;
    params.minInertiaRatio = 0.05;
    params.maxInertiaRatio = 1;

    params.filterByConvexity = false;
    params.filterByColor = true;
    params.blobColor = 255;

//    params.filterByCircularity = true;
//    params.minCircularity = 0.1;

    params.filterByArea = true;
    params.minArea = 0.5f;
//    params.maxArea = 500.0f;

    cv::Ptr<cv::SimpleBlobDetector> d = cv::SimpleBlobDetector::create(params);
    std::vector<cv::KeyPoint> keypoints;

    // Blob detection
    d->detect(imgSrc, keypoints);
    drawKeypoints(imgOriginal, keypoints, imgDst, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DEFAULT);

    imshow("Blob", imgDst);
}

int main() {
    string originalWindow = "Original", thresholdWindow = "Threshold";
    imgOriginal = imread(FILENAME, IMREAD_COLOR); // load image in grayscale

    if (imgOriginal.empty()) {
        cout << "Could not open imgOriginal!" << std::endl;
        return -1;
    }

    // Window with original image.
    namedWindow(originalWindow, WINDOW_AUTOSIZE);
    imshow(originalWindow, imgOriginal);

    // Prepare gamma mask
    Mat imgGammaMask;
    Operations::gammaCorrection(imgOriginal, imgGammaMask, GAMMA_VALUE, false);
    Operations::preprocessImage(imgGammaMask, imgGammaMask, false);
    cv::Mat element = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(2 * elementSize + 1, 2 * elementSize + 1),
            cv::Point(elementSize, elementSize)
    );
    if (blurGammaMask) {
        dilate(imgGammaMask, imgGammaMask, element);
        cv::GaussianBlur(imgGammaMask, imgGammaMask, cv::Size(13, 13), 0);
        if (showAllImages) imshow("Threshold", imgGammaMask);
    }

    // Subtract gamma mask from original image
    cv::cvtColor(imgOriginal, imgOriginal, cv::COLOR_BGR2GRAY);
    subtract(imgOriginal, imgGammaMask, imgSubtracted);
    if (showAllImages) imshow("Subtracted", imgSubtracted);

    // Image binarization
    namedWindow("Binarized");
    createTrackbar("Threshold value", "Binarized", &thresholdValue, 255, updateThreshold);
    createTrackbar("Threshold Block size", "Binarized", &thresholdBlockSize, 255, updateThreshold);
    createTrackbar("Threshold C", "Binarized", &thresholdC, 255, updateThreshold);
    createTrackbar("Median size", "Binarized", &medianSize, 10, updateThreshold);
    createTrackbar("Element size", "Binarized", &elementSize, 20, updateThreshold);

    updateThreshold(0, 0);

    waitKey(0);
    return 0;
}

void updateThreshold(int, void *) {
    cv::Mat element;
    if (thresholdBlockSize % 2 != 1) thresholdBlockSize -= 1;
    if (thresholdBlockSize <= 5) thresholdBlockSize = 5;
    if (medianSize % 2 == 0 || medianSize < 1) medianSize++;
    if (elementSize % 2 == 0 || elementSize < 1) elementSize++;

    if (showAllImages) {
        cout << "threshold block size: " << thresholdBlockSize << ", thresholdC: " << thresholdC << endl;
    }

    // Apply adaptive threshold to original image without nucleus
    adaptiveThreshold(imgSubtracted, imgBinarized, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,
                      thresholdBlockSize, thresholdC);
    medianBlur(imgBinarized, imgBinarized, medianSize);
    if (showAllImages) imshow("Binarized - DT computed from this pic", imgBinarized);

    // Perform the distance transform algorithm
    Mat imgDistanceTransform = Mat(imgOriginal.size(), imgOriginal.type());
    distanceTransform(imgBinarized, imgDistanceTransform, CV_DIST_L2, DIST_MASK_PRECISE);
    normalize(imgDistanceTransform, imgDistanceTransform, 0, 1., NORM_MINMAX);

    // Properly convert distance transform image
    double min, max;
    minMaxLoc(imgDistanceTransform, &min, &max);
    if (min != max) {
        imgDistanceTransform.convertTo(imgDistanceTransform, CV_8U, 255.0/(max-min), -255.0*min/(max-min));
    }
    else {
        // TODO
    }
    imshow("Distance Transform Image", imgDistanceTransform);
    moveWindow("Distance Transform Image", imgOriginal.cols, 1);

    Mat imgDistanceTransformMask;
    threshold(imgDistanceTransform, imgDistanceTransformMask, 40, 255, CV_THRESH_BINARY_INV);
    element = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(elementSize + 1, elementSize + 1),
            cv::Point(elementSize, elementSize)
    );
    if (showAllImages) imshow("Binarized DT", imgDistanceTransformMask);

    erode(imgDistanceTransformMask, imgDistanceTransformMask, element);
    imshow("Inverted binary DT + erosion", imgDistanceTransformMask);
    moveWindow("Inverted binary DT + erosion", imgOriginal.cols, imgOriginal.rows + 1);

    // Subtract DT binary inverted mask from original image
    Mat result = Mat(imgOriginal.size(), imgOriginal.type());
    subtract(imgOriginal, imgDistanceTransformMask, result);
    imshow("result", result);

    blob(result, imgPlaceholder);
}


/* TODO: Do buduceho cvika
 * 1. spocitat pocet jadier
 * 2. aplikovat DT masku na povodny obrazok a spocitat nukleoidy
 * 3. CNN vyskusat
 */

void removeNucleus(cv::Mat &imgSrc, cv::Mat &imgDst, bool showImg) {
    Mat imgTmp;

    // TODO: this a debug blob print
    Operations::simpleBlobDetection(imgSrc, imgTmp, true);

    // Clone original image.
    Mat imgWithoutNucleus = Mat(imgSrc.size(), imgSrc.type());
    imgWithoutNucleus = imgSrc.clone();

    vector<KeyPoint> keypoints = Operations::simpleBlobDetection(imgSrc, imgTmp);
    for (int i = 0; i < keypoints.size(); i++) {
        Point2f kp = keypoints[i].pt;
        double kpSize = keypoints[i].size;
        circle(imgWithoutNucleus, kp, int(kpSize) / 3, Scalar(0, 0, 0), CV_FILLED);
    }

    imgDst = imgWithoutNucleus.clone();
    if (showImg) {
        imshow("Nucleus mask", imgDst);
    }
}
