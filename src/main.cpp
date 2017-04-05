#include "iostream"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv/cv.hpp"
//#include "opencv2/xfeatures2d.hpp"
#include "core/draw.h"
#include "common/utils.h"

using namespace cv;
using namespace std;
//using namespace xfeatures2d;

Mat imgOriginal, imgThreshold, imgEroded, imgBlob;

string FILENAME = "../../data/17_cut.png";
int MAX_THRESHOLD = 255, THRESHOLD = 0, THRESHOLD_TYPE = 8;
double FILTER_MULTIPLIER = 0;
int EROSION_SIZE = 11;

void findNucleus();

void closingAndContours();

vector<KeyPoint> simpleBlobDetection(Mat img);
void gammaCorrection(Mat &src, Mat &dst, float fGamma);

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
    simpleBlobDetection(imgOriginal);
    simpleBlobDetection(imgThreshold);

    // do buduca: vymazat jadra buniek,                         DONE
    Mat imgWithoutNucleus = Mat(imgOriginal.size(), imgOriginal.type());
    imgWithoutNucleus = imgOriginal.clone();

    vector<KeyPoint> keypoints = simpleBlobDetection(imgOriginal);
    for (int i = 0; i < keypoints.size(); i++) {
        Point2f kp = keypoints[i].pt;
        double kpSize = keypoints[i].size;

        circle(imgWithoutNucleus, kp, int(kpSize) / 2, Scalar(0, 0, 0), CV_FILLED);
    }
    imshow("Nucleus mask", imgWithoutNucleus);
    imgWithoutNucleus = Draw::drawAndFilterContours(imgWithoutNucleus, imgWithoutNucleus, 255, Utils::AVERAGE, 0);

    // power-law transformacia na zvyraznenie nukleoidov.       DONE
    Mat imgGamma;
    gammaCorrection(imgWithoutNucleus, imgGamma, 0.65);
    imshow("Gamma", imgGamma);

    // detekcia nukleoidov a priradenie k centroidov            TODO

    // Sharpen img.
    Mat imgSharpened;
    Laplacian(imgWithoutNucleus, imgSharpened, CV_8UC1);
    imshow("Sharpened", imgSharpened);

//    filter2D(imgWithoutNucleus, imgSharpened, -1 , kernel, Point( -1, -1 ), 0, BORDER_DEFAULT );
//    cv::GaussianBlur(frame, image, cv::Size(0, 0), 11);
//    cv::addWeighted(frame, 1.5, image, -0.5, 0, image);*/
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

/*
 * 1. threshold na odburanie zvyskov z original obrazku
 * 2. Potom power-law transformacia
 * 3. Distance transform
 */

void gammaCorrection(Mat &src, Mat &dst, float fGamma) {
    unsigned char lut[256];

    for (int i = 0; i < 256; i++) {
        lut[i] = saturate_cast<uchar>(pow((float) (i / 255.0), fGamma) * 255.0f);
    }

    dst = src.clone();
    const int channels = dst.channels();
    switch (channels) {
        case 1: {
            MatIterator_<uchar> it, end;
            for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
                *it = lut[(*it)];
            break;
        }
        case 3: {
            MatIterator_<Vec3b> it, end;
            for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++) {
                (*it)[0] = lut[((*it)[0])];
                (*it)[1] = lut[((*it)[1])];
                (*it)[2] = lut[((*it)[2])];
            }
            break;
        }
    }
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