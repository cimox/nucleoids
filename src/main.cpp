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

Mat imgOriginal, imgPreprocessed;
string FILENAME = "../../data/samples/14.tiff";

void findNucleus();

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

    // Preprocess image.
    Operations::preprocessImage(imgOriginal, imgPreprocessed);
    imshow("Preprocessed", imgPreprocessed);

    // Find nucleus.
    findNucleus();

    waitKey(0);
    return 0;
}

void findNucleus() {
    Mat imgOriginalGrey, imgBlurred, imgPlaceholder;

    Operations::simpleBlobDetection(imgPreprocessed, imgPlaceholder, true);

    // do buduca: vymazat jadra buniek,                         DONE
    Mat imgWithoutNucleus = Mat(imgOriginal.size(), imgOriginal.type());
    imgWithoutNucleus = imgOriginal.clone();

    vector<KeyPoint> keypoints = Operations::simpleBlobDetection(imgOriginal, imgPlaceholder);
    for (int i = 0; i < keypoints.size(); i++) {
        Point2f kp = keypoints[i].pt;
        double kpSize = keypoints[i].size;
        circle(imgWithoutNucleus, kp, int(kpSize) / 2, Scalar(0, 0, 0), CV_FILLED);
    }
    imshow("Nucleus mask", imgWithoutNucleus);
    imgWithoutNucleus = Draw::drawAndFilterContours(imgWithoutNucleus, imgWithoutNucleus, 255, Utils::AVERAGE, 0);
}

/* TODO: w8
 * 1. threshold na odburanie zvyskov z original obrazku
 * 2. Potom power-law transformacia
 * 3. Distance transform -- vyriesi deliace sa bunky
 */
