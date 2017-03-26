#include "iostream"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv/cv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "core/draw.h"
#include "common/utils.h"

using namespace cv;
using namespace std;
using namespace xfeatures2d;

Mat imgOriginal, imgThreshold, imgEroded, imgBlob;

string FILENAME = "../../data/samples/14.tiff";
int MAX_THRESHOLD = 255, THRESHOLD = 0, THRESHOLD_TYPE = 8;
double FILTER_MULTIPLIER = 0;
int EROSION_SIZE = 11;

void findNucleus();
void closingAndContours();
vector<KeyPoint> simpleBlobDetection(Mat img);

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

    // Preprocess image.
    cvtColor(imgOriginal, imgOriginalGrey, COLOR_BGR2GRAY);
    GaussianBlur(imgOriginalGrey, imgBlurred, Size(11, 11), 0);
    threshold(imgBlurred, imgThreshold, THRESHOLD, MAX_THRESHOLD, THRESH_BINARY + THRESH_OTSU);

    // vyskusat Blob - nastavenie parametrov                    DONE
//    simpleBlobDetection(imgOriginal);
//    simpleBlobDetection(imgThreshold);

    // do buduca: vymazat jadra buniek,                         TODO
    Mat nucleusMask = Mat(imgOriginal.size(), imgOriginal.type());
//    nucleusMask = Scalar(255, 255, 255);

    vector<KeyPoint> keypoints = simpleBlobDetection(imgOriginal);
    for (int i = 0; i < keypoints.size(); i++) {
        Point2f kp = keypoints[i].pt;
        double kpSize = keypoints[i].size;

        circle(nucleusMask, kp, int(kpSize)/3, Scalar(0, 0, 0), CV_FILLED);
        Rect region(int(kp.x), int(kp.y), int(kpSize)/3, int(kpSize)/3);
        GaussianBlur(nucleusMask(region), nucleusMask(region), Size(7, 7), 0);
    }
//    GaussianBlur(nucleusMask, nucleusMask, Size(11, 11), 0);
    imshow("Nucleus mask", nucleusMask);

//    Mat result;
//    imgOriginal.copyTo(result, nucleusMask);
//    imshow("Mask applied", result);

    // power-law transformacia na zvyraznenie nukleoidov.       TODO
    // detekcia nukleoidov a priradenie k centroidov            TODO
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

vector<KeyPoint> simpleBlobDetection(Mat img) {
    SimpleBlobDetector::Params params;
    params.minDistBetweenBlobs = 50.0f;

    params.filterByInertia = true;
    params.minInertiaRatio = 0.5;

    params.filterByConvexity = false;
    params.filterByColor = false;

    params.filterByCircularity = true;
    params.minCircularity = 0.1;

    params.filterByArea = true;
    params.minArea = 250.0f;
    Ptr<SimpleBlobDetector> d = SimpleBlobDetector::create(params);
    vector<KeyPoint> keypoints;

    // Blob detection
    d->detect(img, keypoints);
    drawKeypoints(img, keypoints, imgBlob, Scalar(0, 0, 255), DrawMatchesFlags::DEFAULT);
    imshow("Blob", imgBlob);
    moveWindow("Blob", img.cols, 5);

    return keypoints;
}