#include <iostream>
#include <vector>
#include <memory>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

const int HUE_SLIDER_MAX = 179;
int hue_min_slider = 112;
int hue_max_slider = HUE_SLIDER_MAX;

const int SAT_SLIDER_MAX = 255;
int sat_min_slider = 113;
int sat_max_slider = SAT_SLIDER_MAX;

const int VAL_SLIDER_MAX = 255;
int val_min_slider = 90;
int val_max_slider = VAL_SLIDER_MAX;

const int CONTOUR_AREA_TRESHOLD = 200;

int main(int, char**){
    VideoCapture cap(0);
    if(!cap.isOpened()) return -1;
 
    namedWindow("HSV boundaries", WINDOW_NORMAL);
    namedWindow("capture", WINDOW_NORMAL);
    namedWindow("mask", WINDOW_NORMAL);


    createTrackbar("Hue min", "HSV boundaries", &hue_min_slider, HUE_SLIDER_MAX, nullptr);
    createTrackbar("Hue max", "HSV boundaries", &hue_max_slider, HUE_SLIDER_MAX, nullptr);

    createTrackbar("Satur. min", "HSV boundaries", &sat_min_slider, SAT_SLIDER_MAX, nullptr);
    createTrackbar("Satur. max", "HSV boundaries", &sat_max_slider, SAT_SLIDER_MAX, nullptr);

    createTrackbar("Value min", "HSV boundaries", &val_min_slider, VAL_SLIDER_MAX, nullptr);
    createTrackbar("Value max", "HSV boundaries", &val_max_slider, VAL_SLIDER_MAX, nullptr);

    Mat frame, display_frame, hsv_frame, mask;

    while(true)
    {
        cap >> frame;

        // flip the image to better control our drawing.
        // 1 is a flip flag. Positive value means "flip around Y-axis".
        flip(frame, frame, 1);

        display_frame = frame.clone();

        medianBlur(frame, frame, 5);    // soft blur to reduce noise and keep edges

        // convert RGB image to HSV
        cvtColor(frame, hsv_frame, COLOR_BGR2HSV);

        Scalar lower_bound(hue_min_slider, sat_min_slider, val_min_slider);
        Scalar upper_bound(hue_max_slider, sat_max_slider, val_max_slider);
        
        // create a mask of white pixels fitting in boundaries
        inRange(hsv_frame, lower_bound, upper_bound, mask);


        // kernel for erode and dilate iterations
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));

        // erode - remove "weak" parts of mask
        erode(mask, mask, kernel, Point(-1,-1), 1);

        // dilate - make everything that is left stronger and more consistent
        dilate(mask, mask, kernel, Point(-1,-1), 3);  // more iterations to make pointer stronger

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;

        findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (!contours.empty()){

            double max_area = 0;
            int max_contour_id = -1;

            // find a contour with maximum area
            for (size_t i = 0; i < contours.size(); i++) {
                double area = contourArea(contours[i]);
                if (area > max_area) {
                    max_area = area;
                    max_contour_id = i;
                }
            }

            if (max_contour_id != -1 && max_area > CONTOUR_AREA_TRESHOLD) {
                Moments m = moments(contours[max_contour_id]);

                // calculate it's central coordinates
                if (m.m00 > 0) {
                    int x = m.m10 / m.m00;
                    int y = m.m01 / m.m00;

                    // display a circle on those coordinates
                    circle(display_frame, Point(x, y), 10, Scalar(255, 0, 0), 2);
                }
            }
        }

        imshow("mask", mask);

        imshow("capture", display_frame);
        
        if(waitKey(30) >= 0) break;
    }
    return 0;
}