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

Mat imgOriginal, imgPreprocessed, imgPlaceholder;
string FILENAME = "../../data/samples/14.tiff";

void removeNucleus(cv::Mat &imgSrc, cv::Mat &imgDst, bool showImg = false);

void blob(cv::Mat &imgSrc, cv::Mat &imgDst) {
    cv::SimpleBlobDetector::Params params;
//    params.minDistBetweenBlobs = 50.0f;

    params.filterByInertia = true;
    params.minInertiaRatio = 0.5;

    params.filterByConvexity = false;
    params.filterByColor = false;

    params.filterByCircularity = false;
//    params.minCircularity = 0.1;

    params.filterByArea = true;
    params.minArea = 5.0f;
    params.maxArea = 500.0f;
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

//    bitwise_not(imgOriginal, imgOriginal);
//    cv::GaussianBlur(imgOriginal, imgPlaceholder, cv::Size(0, 0), 3);
//    cv::addWeighted(imgOriginal, 1.5, imgPlaceholder, -0.5, 1, imgPlaceholder);
//    bitwise_not(imgPlaceholder, imgPlaceholder);
//    Operations::gammaCorrection(imgPlaceholder, imgPlaceholder, 0.65);
//    imshow("Laplac", imgPlaceholder);
//    Mat tmp;
//    blob(imgPlaceholder, tmp);

    // Preprocess image.
//    Operations::preprocessImage(imgOriginal, imgPreprocessed);

    // removeNucleus nucleus.
    Mat tmp;
    Operations::gammaCorrection(imgOriginal, imgPlaceholder, 6.5, true);
    bitwise_not(imgPlaceholder, imgPlaceholder);
    cv::cvtColor(imgPlaceholder, imgPlaceholder, cv::COLOR_BGR2GRAY);
    threshold(imgPlaceholder, imgPlaceholder, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    imshow("bitwise", imgPlaceholder);
    imgOriginal.copyTo(imgOriginal, imgPlaceholder);
    imshow("test", imgOriginal);

//    removeNucleus(imgOriginal, imgPlaceholder, true);

    waitKey(0);
    return 0;
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

/* TODO: w8
 * 1. threshold na odburanie zvyskov z original obrazku
 * 2. Potom power-law transformacia
 * 3. Distance transform -- vyriesi deliace sa bunky
 */
