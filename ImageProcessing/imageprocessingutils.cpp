#include "imageprocessingutils.h"

#include <cmath>

#include <opencv2/highgui/highgui.hpp>

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
