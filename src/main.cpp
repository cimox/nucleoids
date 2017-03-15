#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

int main() {
    Mat image;
    image = imread("../../data/5-6/005-Z_cut.tif", IMREAD_COLOR);

    if (image.empty()) {
        cout << "Could not open image!" << std::endl;
        return -1;
    }

    std::cout << image.at<int>(0, 1);

    namedWindow("Test image", WINDOW_AUTOSIZE);
    imshow("Test image", image);

    waitKey(0);
    return 0;
}