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

// Image matrices
Mat imgOriginal, imgPlaceholder, imgSubtracted, imgBinarized;

// Filenames
string FILENAME = "../../data/samples/31_mid-brightness.tiff";
string WRITE_FILENAME = "../../data/results/tmp.png";

// Constants
int GAMMA_VALUE = 10, ADAPTIVE_MAX_THRESHOLD_VALUE = 255, ADAPTIVE_THRESHOLD_BLOCK_SIZE = 5, ADAPTIVE_THRESHOLD_C = 0,
        MEDIAN_SIZE = 7, STRUCT_ELEM_SIZE = 7, SUBTRACT_ELEM_SIZE = 3;

// Nucleoids positions keypoints from blob detection
std::vector<cv::KeyPoint> nucleoidsPositions, nucleiPositions;

// Debug variables
bool DEBUG = false, SHOW_ORIGINAL = false;


void removeNucleus(cv::Mat &imgSrc, cv::Mat &imgDst, bool showImg = false);

void findNucleoids(int, void *);

std::vector<cv::KeyPoint> findNucleoidBlobs(cv::Mat &imgSrc, cv::Mat imgDstSource, cv::Mat &imgDst, bool showImg) {
    cv::SimpleBlobDetector::Params params;

    params.minDistBetweenBlobs = 10.0f;

    params.filterByInertia = true;
    params.minInertiaRatio = 0.05;
    params.maxInertiaRatio = 1;

    params.filterByConvexity = false;
    params.filterByColor = true;
    params.blobColor = 255;

    params.filterByArea = true;
    params.minArea = 0.01f;

    cv::Ptr<cv::SimpleBlobDetector> d = cv::SimpleBlobDetector::create(params);
    std::vector<cv::KeyPoint> keypoints;

    // Blob detection
    d->detect(imgSrc, keypoints);
    drawKeypoints(imgDstSource, keypoints, imgDst, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DEFAULT);

    if (showImg) {
        string label = to_string(keypoints.size()) + " nucleoids found";
        putText(imgDst, label, cv::Point(50, 50), cv::QT_FONT_NORMAL, 2.0, CV_RGB(0, 204, 0), 2);

        Mat imgTmp;
        if (imgDst.rows / 2 > 1024 || imgDst.cols / 2 > 800) {
            resize(imgDst, imgTmp, Size(imgDst.rows / 2, imgDst.cols / 2));
        }

        imshow("Nucleoids", imgTmp);
    }

    return keypoints;
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
    if (DEBUG || SHOW_ORIGINAL) imshow(originalWindow, imgOriginal);

    // Prepare gamma mask
    Mat imgGammaMask;
    Operations::gammaCorrection(imgOriginal, imgGammaMask, GAMMA_VALUE, false);
    Operations::preprocessImage(imgGammaMask, imgGammaMask, false);
    cv::Mat element = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(2 * SUBTRACT_ELEM_SIZE + 1, 2 * SUBTRACT_ELEM_SIZE + 1),
            cv::Point(SUBTRACT_ELEM_SIZE, SUBTRACT_ELEM_SIZE)
    );
    dilate(imgGammaMask, imgGammaMask, element);
    if (DEBUG) imshow("Threshold", imgGammaMask);

    // Count cell nuclei
    nucleiPositions = Operations::countNucleus(imgGammaMask, imgOriginal, "Cell nuclei count", true);

    // Subtract gamma mask from original image
    cv::cvtColor(imgOriginal, imgOriginal, cv::COLOR_BGR2GRAY);
    subtract(imgOriginal, imgGammaMask, imgSubtracted);
    if (DEBUG) imshow("Subtracted", imgSubtracted);

    // Nucleoids segmentation -- core
    if (DEBUG) {
        namedWindow("Trackbars");
        createTrackbar("Threshold value", "Trackbars", &ADAPTIVE_MAX_THRESHOLD_VALUE, 255, findNucleoids);
        createTrackbar("Threshold Block size", "Trackbars", &ADAPTIVE_THRESHOLD_BLOCK_SIZE, 255, findNucleoids);
        createTrackbar("Threshold C", "Trackbars", &ADAPTIVE_THRESHOLD_C, 255, findNucleoids);
        createTrackbar("Median size", "Trackbars", &MEDIAN_SIZE, 10, findNucleoids);
        createTrackbar("Element size", "Trackbars", &STRUCT_ELEM_SIZE, 20, findNucleoids);
    }

    findNucleoids(0, 0);

    waitKey(0);
    return 0;
}


Mat getImageWithNucleoidsAreas() {
    cv::Mat element;
    // Apply adaptive threshold to original image without nucleus
    adaptiveThreshold(imgSubtracted, imgBinarized, ADAPTIVE_MAX_THRESHOLD_VALUE, CV_ADAPTIVE_THRESH_MEAN_C,
                      CV_THRESH_BINARY,
                      ADAPTIVE_THRESHOLD_BLOCK_SIZE, ADAPTIVE_THRESHOLD_C);
    medianBlur(imgBinarized, imgBinarized, MEDIAN_SIZE);
    if (DEBUG) imshow("1. Binarized subtracted image (adaptive threshold + median blur)", imgBinarized);

    // Perform the distance transform algorithm
    Mat imgDistanceTransform = Mat(imgOriginal.size(), imgOriginal.type());
    distanceTransform(imgBinarized, imgDistanceTransform, CV_DIST_L2, DIST_MASK_PRECISE);
    normalize(imgDistanceTransform, imgDistanceTransform, 0, 1., NORM_MINMAX);

    // Properly convert distance transform image
    double min, max;
    minMaxLoc(imgDistanceTransform, &min, &max);
    if (min != max) {
        imgDistanceTransform.convertTo(imgDistanceTransform, CV_8U, 255.0 / (max - min), -255.0 * min / (max - min));
    }
    medianBlur(imgDistanceTransform, imgDistanceTransform, MEDIAN_SIZE);
    if (DEBUG) {
        imshow("2. DT of binarized image (DT, normalization, median blur)", imgDistanceTransform);
        moveWindow("2. DT of binarized image (DT, normalization, median blur)", imgOriginal.cols, 1);
    }

    Mat imgDistanceTransformMask;
    threshold(imgDistanceTransform, imgDistanceTransformMask, 40, 255, CV_THRESH_BINARY_INV);
    element = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(STRUCT_ELEM_SIZE + 1, STRUCT_ELEM_SIZE + 1),
            cv::Point(STRUCT_ELEM_SIZE, STRUCT_ELEM_SIZE)
    );
    erode(imgDistanceTransformMask, imgDistanceTransformMask, element);
    if (DEBUG) imshow("3. Inverted DT image (binary inv threshold + erosion)", imgDistanceTransformMask);

    // Subtract DT binary inverted mask from original image
    Mat imgWithNucleoidsAreas = Mat(imgOriginal.size(), imgOriginal.type());
    subtract(imgOriginal, imgDistanceTransformMask, imgWithNucleoidsAreas);
    if (DEBUG) imshow("4. Original - inverted eroded mask", imgWithNucleoidsAreas);

    return imgWithNucleoidsAreas;
}

void clampVariables() {
    if (ADAPTIVE_THRESHOLD_BLOCK_SIZE % 2 != 1) ADAPTIVE_THRESHOLD_BLOCK_SIZE -= 1;
    if (ADAPTIVE_THRESHOLD_BLOCK_SIZE <= 5) ADAPTIVE_THRESHOLD_BLOCK_SIZE = 5;
    if (MEDIAN_SIZE % 2 == 0 || MEDIAN_SIZE < 1) MEDIAN_SIZE++;
    if (STRUCT_ELEM_SIZE % 2 == 0 || STRUCT_ELEM_SIZE < 1) STRUCT_ELEM_SIZE++;
}

void findNucleoids(int, void *) {
    cv::Mat element;

    clampVariables();

    if (DEBUG) {
        cout << "elem: " << STRUCT_ELEM_SIZE << ", median: " << MEDIAN_SIZE << endl;
        cout << "threshold block size: " << ADAPTIVE_THRESHOLD_BLOCK_SIZE << ", ADAPTIVE_THRESHOLD_C: "
             << ADAPTIVE_THRESHOLD_C << endl;
    }

    Mat imgWithNucleoidsAreas = getImageWithNucleoidsAreas();

    // DT of original image where only areas with nucleoids are kept
    Mat imgResultThreshold;
    adaptiveThreshold(imgWithNucleoidsAreas, imgResultThreshold, ADAPTIVE_MAX_THRESHOLD_VALUE,
                      CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY,
                      ADAPTIVE_THRESHOLD_BLOCK_SIZE, ADAPTIVE_THRESHOLD_C);
    if (DEBUG) imshow("Result threshold", imgResultThreshold);

    // Apply DT on binarized imgWithNucleoidsAreas
    Mat imgDTResult = Mat(imgOriginal.size(), imgOriginal.type());
    distanceTransform(imgResultThreshold, imgDTResult, CV_DIST_L2, DIST_MASK_PRECISE);
    normalize(imgDTResult, imgDTResult, 0, 1., NORM_MINMAX);

    // Properly convert distance transform image
    double min, max;
    minMaxLoc(imgDTResult, &min, &max);
    if (min != max) {
        imgDTResult.convertTo(imgDTResult, CV_8U, 255.0 / (max - min), -255.0 * min / (max - min));
    }
    medianBlur(imgDTResult, imgDTResult, 1);
    if (DEBUG) {
        imshow("DT of binarized imgWithNucleoidsAreas", imgDTResult);
        findNucleoidBlobs(imgDTResult, imgDTResult, imgPlaceholder, true);
    }

    nucleoidsPositions = findNucleoidBlobs(imgDTResult, imgOriginal, imgPlaceholder, false);

    // Assign nucleoids to nuclei -- TODO: extract this part to separate function
    unsigned int nucleoidsWithNucleusCount = 0;
    for (vector<KeyPoint>::iterator nucleoid = nucleoidsPositions.begin(); nucleoid != nucleoidsPositions.end();) {
        double shortestLine = DBL_MAX;
        Point bestNucleusCenter = Point(-1, -1);

        for (auto nucleus : nucleiPositions) {
            if (pow((nucleoid->pt.x - nucleus.pt.x), 2) + pow((nucleoid->pt.y - nucleus.pt.y), 2) <
                pow(nucleus.size * 1.5, 2)) {
                if (norm(nucleus.pt - nucleoid->pt) < shortestLine) {
                    bestNucleusCenter = nucleus.pt;
                    shortestLine = norm(nucleus.pt - nucleoid->pt);
                }
            }
        }

        if (bestNucleusCenter.x != -1 && bestNucleusCenter.y != -1) {
            line(imgPlaceholder, bestNucleusCenter, nucleoid->pt, Scalar(0, 255, 0), 1);
            nucleoidsWithNucleusCount++;
        }
        ++nucleoid;
    }

    string label = to_string(nucleoidsWithNucleusCount) + " nucleoids found";
    putText(imgPlaceholder, label, cv::Point(imgPlaceholder.cols / 2, 50), cv::QT_FONT_NORMAL, 1.0,
            CV_RGB(0, 204, 0), 1);
    if (imgPlaceholder.rows / 2 > 1024 || imgPlaceholder.cols / 2 > 800) {
        resize(imgPlaceholder, imgPlaceholder, Size(imgPlaceholder.rows / 2, imgPlaceholder.cols / 2));
    }
    imwrite(WRITE_FILENAME, imgPlaceholder);

    imshow("Result", imgPlaceholder);
}

void removeNucleus(cv::Mat &imgSrc, cv::Mat &imgDst, bool showImg) {
    Mat imgTmp;

    // TODO: this a debug blob print
    Operations::simpleBlobDetection(imgSrc, imgTmp, "Remove nucleus - blob", true);

    // Clone original image.
    Mat imgWithoutNucleus = Mat(imgSrc.size(), imgSrc.type());
    imgWithoutNucleus = imgSrc.clone();

    vector<KeyPoint> keypoints = Operations::simpleBlobDetection(imgSrc, imgTmp, "Removed nucleus");
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
