#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

Mat imgOriginal, imgThreshold;
Mat nucleusPositions;

string originalWindow = "Original", thresholdWindow = "Threshold";
string trackbarThresholdValue = "Threshold value", trackbarThresholdType = "Threshold type";
int maxThresholdValue = 255, thresholdValue = 0,
        maxThresholdType = 8, thresholdType = 8;
RNG rng(12345);

void thresholdCallback(int, void*);
void showNucleusPositions(Mat imgThreshold);
void showNucleusContours(Mat imgOriginal, Mat imgThreshold, int thresholdValue);

int main() {
    imgOriginal = imread("../../data/17_cut.png", IMREAD_COLOR); // load image in grayscale

    if (imgOriginal.empty()) {
        cout << "Could not open imgOriginal!" << std::endl;
        return -1;
    }

    // Window with original image.
    namedWindow(originalWindow, WINDOW_AUTOSIZE);
    imshow(originalWindow, imgOriginal);

    // Window with image after threshold.
    namedWindow(thresholdWindow, WINDOW_AUTOSIZE);
    createTrackbar(trackbarThresholdValue, originalWindow, &thresholdValue, maxThresholdValue, thresholdCallback);
    createTrackbar(trackbarThresholdType, originalWindow, &thresholdType, maxThresholdType, thresholdCallback);

    waitKey(0);
    return 0;
}

void thresholdCallback(int, void*)
{
    imgOriginal = imread("../../data/17_cut.png", IMREAD_COLOR); // reload image in grayscale

    cout  << "threshold " << thresholdType << ", value: " << thresholdValue << endl;
    if (thresholdType == 8) {
        thresholdType = THRESH_BINARY | thresholdType;
    }

    Mat imgOriginalGrey;
    cvtColor(imgOriginal, imgOriginalGrey, COLOR_BGR2GRAY);

    threshold(imgOriginalGrey, imgThreshold, thresholdValue, 255, thresholdType);
    imshow(thresholdWindow, imgThreshold);

//    showNucleusPositions(imgThreshold);
    showNucleusContours(imgOriginal, imgThreshold, thresholdValue);
}

void showNucleusPositions(Mat imgThreshold)
{
    vector<Point> nucleusPositions;
    findNonZero(imgThreshold, nucleusPositions);

    cout << "Nucleus positions: " << nucleusPositions << endl;
}

double calculateMedian(vector<double> values)
{
    double median;
    size_t size = values.size();

    sort(values.begin(), values.end());

    if (size  % 2 == 0)
        median = (values[size / 2 - 1] + values[size / 2]) / 2;
    else
        median = values[size / 2];

    return median;
}

vector<vector<Point>> filterContours(vector<vector<Point>> contours)
{
    double medianAreaSize = 0;
    vector<double> contourAreas;

    for( int i = 0; i < contours.size(); i++ )
    {
        contourAreas.push_back(contourArea(contours[i]));
    }
    medianAreaSize = calculateMedian(contourAreas);
    cout << "Median of contour area size " << medianAreaSize << endl;

    cout << contours.size() << endl;
    for ( int i = 0; i < contours.size(); i++)
    {
        contours.erase(contours.begin() + i);
//        if (contourArea(contours[i]) <= 50000.0)
//        {
//            contours.erase(contours.begin() + i);
//        }
    }
    cout << contours.size() << endl;

    return contours;
}

void showNucleusContours(Mat imgOriginal, Mat imgThreshold, int thresholdValue)
{
    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    /// Detect edges using canny
    Canny( imgThreshold, canny_output, thresholdValue, thresholdValue*2, 3 );
    /// Find contours
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    cout << "Filtering " << contours.size() << " contours" << endl;
    contours = filterContours(contours);
    cout << "Contours " << contours.size() << " left" << endl;

    /// Draw contours
    for( int i = 0; i< contours.size(); i++ )
    {
        drawContours( imgOriginal, contours, i, (255, 255, 255), 2, 8, hierarchy, 0, Point() );
    }

    /// Show in a window
    namedWindow( originalWindow, CV_WINDOW_AUTOSIZE );
    imshow( originalWindow, imgOriginal );
}
