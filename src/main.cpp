#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

Mat imgOriginal, imgThreshold;
Mat nucleusPositions;

string filename = "../../data/5-6/005-Z_cut.tif";
string originalWindow = "Original", thresholdWindow = "Threshold";
string trackbarThresholdValue = "Threshold value", trackbarThresholdType = "Threshold type";
int maxThresholdValue = 255, thresholdValue = 0,
        maxThresholdType = 8, thresholdType = 8;
RNG rng(12345);

void thresholdCallback(int, void*);
void showNucleusPositions(Mat imgThreshold);
void showNucleusContours(Mat imgOriginal, Mat imgThreshold, int thresholdValue);

int main() {
    imgOriginal = imread(filename, IMREAD_COLOR); // load image in grayscale

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

double calculateVectorMedian(vector<double> values)
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

double calculateVectorAverage(vector<double> values)
{
    double average = 0;

    for (int i = 0; i < values.size(); i++)
    {
        average += values[i];
    }

    return average/values.size();
}

vector<double> getContourAreas(vector<vector<Point>> contours)
{
    vector<double> contourAreas;
    for( int i = 0; i < contours.size(); i++ )
    {
        contourAreas.push_back(contourArea(contours[i]));
    }

    return contourAreas;
}

void func ( void (*f)(int) );
vector<vector<Point>> filterContours(vector<vector<Point>> contours, double (*metricFunction)(vector<double>) )
{
    vector<double> contourAreas = getContourAreas(contours);
    double medianAreaSize = metricFunction(contourAreas);

    cout << "Median of contour area size " << medianAreaSize << endl;

    if (contours.size() <= 1)
    {
        return contours;
    }

    for ( long i = contours.size() - 1; i >= 0; i--)
    {
        if (contourArea(contours[i]) < medianAreaSize)
        {
            if (contours.size() == 1)
            {
                contours.erase(contours.begin());
            }
            else
            {
                contours.erase(contours.begin() + i);
            }
        }
    }

    return contours;
}

void showNucleusContours(Mat imgOriginal, Mat imgThreshold, int thresholdValue)
{
    Mat canny_output;
    Mat imgOriginalContours;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    imgOriginal.copyTo(imgOriginalContours);
    Canny( imgThreshold, canny_output, thresholdValue, thresholdValue*2, 3 );
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    cout << "Filtering " << contours.size() << " contours" << endl;
    contours = filterContours(contours, calculateVectorAverage);
    cout << "Contours " << contours.size() << " left" << endl;

    for( int i = 0; i< contours.size(); i++ )
    {
        drawContours( imgOriginalContours, contours, i, (255, 255, 255), 2, 8, hierarchy, 0, Point() );
    }

    namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    imshow( "Contours", imgOriginalContours );
}
