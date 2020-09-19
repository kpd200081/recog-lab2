#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

void blur(const Mat& source, Mat& result, int k_size) {
    Mat kernel(k_size, k_size, CV_64F);
    for(int i = 0; i < k_size; i++) {
        for(int j = 0; j < k_size; j++) {
            kernel.at<double>(i, j) = double(1 / pow(k_size, 2));
        }
    }
    filter2D(source, result, source.depth(), kernel);
}

void custom_sobel(const Mat& source, Mat& result) {
    Mat kernel(3, 3, CV_16S);

    // from up to down
    kernel.at<short>(0, 0) = -1;
    kernel.at<short>(0, 1) = -2;
    kernel.at<short>(0, 2) = -1;

    kernel.at<short>(1, 0) = 0;
    kernel.at<short>(1, 1) = 0;
    kernel.at<short>(1, 2) = 0;

    kernel.at<short>(2, 0) = 1;
    kernel.at<short>(2, 1) = 2;
    kernel.at<short>(2, 2) = 1;

    filter2D(source, result, source.depth(), kernel);
}

int main() {
    Mat src = imread("../download.jpeg", IMREAD_COLOR);

    if(!src.data) {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    namedWindow("Source", WINDOW_AUTOSIZE);
    imshow("Source", src);

    Mat gblur;
    GaussianBlur(src, gblur, Size(13, 7), 0, 0);
    namedWindow("Gaus Blur", WINDOW_AUTOSIZE);
    imshow("Gaus Blur", gblur);

    Mat cblur;
    blur(src, cblur, 3);
    namedWindow("Custom Blur", WINDOW_AUTOSIZE);
    imshow("Custom Blur", cblur);

    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    Mat sobel;
    custom_sobel(gray, sobel);
    namedWindow("Custom Sobel", WINDOW_AUTOSIZE);
    imshow("Custom Sobel", sobel);

    waitKey(0);
    return 0;
}
