//
// Created by Matúš Cimerman on 05/04/2017.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv/cv.hpp>

namespace Operations {

    int MAX_THRESHOLD = 255, THRESHOLD = 0;

    void preprocessImage(cv::Mat &imgSrc, cv::Mat &imgDst, bool showImg) {
        cv::Mat imgTmp;
        cv::cvtColor(imgSrc, imgTmp, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(imgTmp, imgTmp, cv::Size(11, 11), 0);
        cv::threshold(imgTmp, imgDst, THRESHOLD, MAX_THRESHOLD, cv::THRESH_BINARY + cv::THRESH_OTSU);

        if (showImg) {
            imshow("Preprocessed", imgDst);
        }
    }

    void morphClosing(cv::Mat &imgSrc, cv::Mat &imgDst, int erosion_size, bool showImg) {
        // Erosion + dilatation.
        cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                            cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                            cv::Point(erosion_size, erosion_size));
        erode(imgSrc, imgDst, element);
        dilate(imgDst, imgDst, element);
        if (showImg) {
            imshow("Morphological closing", imgDst);
        }
    }

    void gammaCorrection(cv::Mat &imgSrc, cv::Mat &imgDst, float fGamma, bool showImg) {
        unsigned char lut[256];

        for (int i = 0; i < 256; i++) {
            lut[i] = cv::saturate_cast<uchar>(pow((float) (i / 255.0), fGamma) * 255.0f);
        }

        imgDst = imgSrc.clone();
        const int channels = imgDst.channels();
        switch (channels) {
            case 1: {
                cv::MatIterator_<uchar> it, end;
                for (it = imgDst.begin<uchar>(), end = imgDst.end<uchar>(); it != end; it++)
                    *it = lut[(*it)];
                break;
            }
            case 3: {
                cv::MatIterator_<cv::Vec3b> it, end;
                for (it = imgDst.begin<cv::Vec3b>(), end = imgDst.end<cv::Vec3b>(); it != end; it++) {
                    (*it)[0] = lut[((*it)[0])];
                    (*it)[1] = lut[((*it)[1])];
                    (*it)[2] = lut[((*it)[2])];
                }
                break;
            }
            default: break;
        }

        if (showImg) {
            imshow("Gamma corrected", imgDst);
        }
    }

    std::vector<cv::KeyPoint> simpleBlobDetection(cv::Mat &imgSrc, cv::Mat &imgDst, bool showImg) {
        cv::SimpleBlobDetector::Params params;
        params.minDistBetweenBlobs = 50.0f;

        params.filterByInertia = true;
        params.minInertiaRatio = 0.5;

        params.filterByConvexity = false;
        params.filterByColor = false;

        params.filterByCircularity = true;
        params.minCircularity = 0.1;

        params.filterByArea = true;
        params.minArea = 250.0f;
        cv::Ptr<cv::SimpleBlobDetector> d = cv::SimpleBlobDetector::create(params);
        std::vector<cv::KeyPoint> keypoints;

        // Blob detection
        d->detect(imgSrc, keypoints);
        drawKeypoints(imgSrc, keypoints, imgDst, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DEFAULT);

        if (showImg) {
            imshow("Blob", imgDst);
        }

        return keypoints;
    }

}