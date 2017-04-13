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

Mat imgOriginal, imgPreprocessed, imgPlaceholder, imgSubtracted;
string FILENAME = "../../data/samples/14.tiff";
int GAMMA_VALUE = 10;
int thresholdValue = 0, thresholdBlockSize = 5, thresholdC = 0;

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
    params.minArea = 2.5f;
//    params.maxArea = 500.0f;

    cv::Ptr<cv::SimpleBlobDetector> d = cv::SimpleBlobDetector::create(params);
    std::vector<cv::KeyPoint> keypoints;

    // Blob detection
    d->detect(imgSrc, keypoints);
    drawKeypoints(imgSrc, keypoints, imgDst, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DEFAULT);

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
    Operations::gammaCorrection(imgOriginal, imgGammaMask, GAMMA_VALUE, true);
    Operations::preprocessImage(imgGammaMask, imgGammaMask, true);
    int erosion_size = 1;
    cv::Mat element = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
            cv::Point(erosion_size, erosion_size)
    );
    dilate(imgGammaMask, imgGammaMask, element);
    cv::GaussianBlur(imgGammaMask, imgGammaMask, cv::Size(13, 13), 0);
    imshow("Threshold", imgGammaMask);

    // Subtract gamma mask from original image
    cv::cvtColor(imgOriginal, imgOriginal, cv::COLOR_BGR2GRAY);
    subtract(imgOriginal, imgGammaMask, imgSubtracted);
    imshow("Subtracted", imgSubtracted);

    // Image binarization
    Mat imgBinarized;
    int medianSize = 5;
    adaptiveThreshold(imgSubtracted, imgBinarized, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,
                      thresholdBlockSize, thresholdC);
    medianBlur(imgBinarized, imgBinarized, medianSize);
    imshow("Binarized", imgBinarized);

    // Distance transform
    Mat kernel = Mat_<float>(3, 3) << (1, 1, 1, 1, -8, 1, 1, 1, 1);
    Mat imgLaplacian;
    Mat sharp = imgBinarized; // copy source image to another temporary one
    filter2D(sharp, imgLaplacian, CV_32F, kernel);
    imgBinarized.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    // imshow( "Laplace Filtered Image", imgLaplacian );
    imshow("New Sharped Image", imgResult);

    // Perform the distance transform algorithm
    Mat dist;
    distanceTransform(imgBinarized, dist, CV_DIST_L2, 3);
    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(dist, dist, 0, 1., NORM_MINMAX);
    imshow("Distance Transform Image", dist);

    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    threshold(dist, dist, .4, 1., CV_THRESH_BINARY);
    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
    dilate(dist, dist, kernel1);
    imshow("Peaks", dist);


    // Try remove contours and blob detection -- not working properly
//    Mat imgPlaceholder = imgBinarized.clone();
//    Draw::drawAndFilterContours(imgPlaceholder, imgBinarized, 5, Utils::AVERAGE, 1);
//    blob(imgBinarized, imgPlaceholder);
//    imshow("blob", imgPlaceholder);

//    removeNucleus(imgOriginal, imgPlaceholder, true);
    waitKey(0);
    return 0;
}

void updateThreshold(int, void *) {
    //    namedWindow("Binarized");
//    createTrackbar("Threshold value", "Binarized", &thresholdValue, 255, updateThreshold);
//    createTrackbar("Threshold Block size", "Binarized", &thresholdBlockSize, 255, updateThreshold);
//    createTrackbar("Threshold C", "Binarized", &thresholdC, 255, updateThreshold);
//
//    updateThreshold(0, 0);

    Mat imgBinarized;
//    threshold(imgSubtracted, imgBinarized, thresholdValue, 255, THRESH_OTSU);
    if (thresholdBlockSize % 2 != 1) thresholdBlockSize -= 1;
    if (thresholdBlockSize <= 5) thresholdBlockSize = 5;
    adaptiveThreshold(imgSubtracted, imgBinarized, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,
                      thresholdBlockSize, thresholdC);

    imshow("Binarized", imgBinarized);
}

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
