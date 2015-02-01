#include "imageprocessingutils.h"

#include <cmath>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;

#define D_DEBUG 0

// from OpenCV Samples
// finds a cosine of angle between vectors from pt0->pt1 and from pt0->pt2
double image_processing_utils::findAngle(Point p1, Point p2, Point p0)
{
    double dx1 = p1.x - p0.x;
    double dy1 = p1.y - p0.y;
    double dx2 = p2.x - p0.x;
    double dy2 = p2.y - p0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// from OpenCV Samples
// returns sequence of squares detected on the image, with a specific size in ratio.
std::vector<std::vector<cv::Point>> image_processing_utils::findSquares(const Mat& image, double minAreaPercentageFilter, double maxAreaPercentageFilter, double wholeArea, const int thresh, const int N)
{
	std::vector<std::vector<Point>> squares;

    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for (int c = 0; c < 3; ++c)
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for (int l = 0; l < N; ++l)
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if (l == 0)
            {
                // apply Canny. Take the upper threshold from slider and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
                // apply threshold if l!=0: tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;

            // find contours and store them all as a list
            findContours(gray, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            vector<Point> approx;
            for (auto& contour : contours)
            {
                // approximate contour with accuracy proportional to the contour perimeter
                approxPolyDP(Mat(contour), approx, arcLength(Mat(contour), true)*0.02, true);

                // square contours should have 4 vertices after approximation relatively large area (to filter out noisy contours) and be convex.
                double area = 100.0*fabs(contourArea(Mat(approx)))/wholeArea;
                if (approx.size() == 4 && area >= minAreaPercentageFilter && area <= maxAreaPercentageFilter && isContourConvex(Mat(approx)))
                {
                    double maxCosine = 0;
                    for ( int j = 2; j < 5; j++ )
                        // find the maximum cosine of the angle between joint edges
                        maxCosine = std::max(maxCosine, fabs(image_processing_utils::findAngle(approx[j%4], approx[j-2], approx[j-1])));

                    // if cosines of all angles are small (all angles are ~90 degree) then write quandrange vertices to resultant sequence
                    if (maxCosine < 0.3)
                        squares.push_back(approx);
                }
            }
        }
    }
    return squares;
}

void image_processing_utils::drawSquares(Mat& image, const std::vector<std::vector<Point>>& squares, bool R, bool G, bool B)
{
	for (auto& square : squares)
    {
        const Point* p = &square[0];
        int n = (int)square.size();
        char r,g,b;
        r = g = b = 0;
        if(R)
        	r = 255;
        else if(G)
        	g = 255;
        else if(B)
        	b = 255;
        else
        	r = g = b = 128;

        polylines(image, &p, &n, 1, true, Scalar(r,g,b), 3, CV_AA);
    }
}
