#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

void filter(const Mat& src, Mat& dst, const Mat& kernel) {
    if(kernel.size[0] != kernel.size[1]) {
        cerr << "Only for square kernel!" << endl;
        exit(-1);
    }
    if(kernel.size[0] % 2 == 0) {
        cerr << "Only support odd kernel size!" << endl;
        exit(-1);
    }
    if(src.type() != CV_8U && src.type() != CV_8UC1 && src.type() != CV_8UC2 &&
       src.type() != CV_8UC3 && src.type() != CV_8UC4) {
        cerr << "Only support 8UC1 or 8UC3 matrixes!" << endl;
        exit(-1);
    }
    int k_size = int(floor(kernel.size[0] / 2));

    // conver kernel to 3 channels
    Mat loc_kernel;
    kernel.convertTo(loc_kernel, CV_64F);
    if(kernel.type() != src.type()) {
        vector<Mat> arr;
        for(int i = 0; i < src.channels(); i++) {
            arr.push_back(loc_kernel);
        }
        merge(arr.data(), size_t(src.channels()), loc_kernel);
    }

    // add zeros to Mat
    Mat temp;
    src.convertTo(temp, CV_64F);
    Mat temp2(src.size[0] + k_size * 2, src.size[1] + k_size * 2, temp.type());
    Mat cols = Mat::zeros(src.size[0], k_size, temp.type());
    Mat rows = Mat::zeros(k_size, src.size[1] + k_size * 2, temp.type());
    hconcat(cols, temp, temp2);
    temp = temp2;
    hconcat(temp, cols, temp2);
    temp = temp2;
    vconcat(rows, temp, temp2);
    temp = temp2;
    vconcat(temp, rows, temp2);
    temp = temp2;
    dst = Mat(src.size[0], src.size[1], CV_8UC3);

    // for matrix internal
    Mat res;
    for(int i = k_size; i < temp.size[0] - k_size; i++) {
        for(int j = k_size; j < temp.size[1] - k_size; j++) {

            temp2 = temp(Rect(j - k_size, i - k_size, loc_kernel.size[0], loc_kernel.size[0]));
            Mat res = Mat::zeros(1, 1, temp.type());
            for(int k = 0; k < loc_kernel.size[0]; k++) {
                for(int l = 0; l < loc_kernel.size[0]; l++) {
                    res += temp2(Rect(k, l, 1, 1)).mul(loc_kernel(Rect(k, l, 1, 1)));
                }
            }
            res.convertTo(res, src.type());
            if(res.channels() > 1) {
                for(int c = 0; c < res.channels(); c++) {
                    dst.ptr(i - k_size, j - k_size)[c] = res.ptr(0, 0)[c];
                }
            } else {
                // don't know why this hack required, but this doesn't corrupt image
                Mat mask = Mat::zeros(src.size[0], src.size[1], CV_8UC1);
                mask.at<uchar>(i - k_size, j - k_size) = 1;
                dst.setTo(res, mask);
            }
        }
    }
    dst.convertTo(dst, src.type());
}

void blur(const Mat& source, Mat& result, int k_size) {
    Mat kernel(k_size, k_size, CV_64F, Scalar(1 / pow(k_size, 2)));
    filter(source, result, kernel);
}

void custom_sobel(const Mat& source, Mat& result) {
    // from up to down
    short data[] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
    Mat kernel(3, 3, CV_16S, data);

    filter(source, result, kernel);
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
    blur(src, cblur, 5);
    namedWindow("Custom Blur", WINDOW_AUTOSIZE);
    imshow("Custom Blur", cblur);

    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_8UC1);

    Mat sobel;
    custom_sobel(gray, sobel);
    namedWindow("Custom Sobel", WINDOW_AUTOSIZE);
    imshow("Custom Sobel", sobel);

    waitKey(0);
    return 0;
}
