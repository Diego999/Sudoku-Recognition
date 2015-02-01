#ifndef IMAGE_PROCESSING_UTILS_H
#define IMAGE_PROCESSING_UTILS_H

#include <vector>
#include <string>
#include <opencv2/highgui/highgui.hpp>

namespace image_processing_utils
{
	double findAngle(cv::Point p1, cv::Point p2, cv::Point p0);
	std::vector<std::vector<cv::Point>> findSquares(const cv::Mat& image, double minAreaPercentageFilter, double maxAreaPercentageFilter, double wholeArea, const int thresh = 50, const int N = 11);
	cv::Mat captureSoduku(const int minAreaPercentageFilter, const int maxAreaPercentageFilter, std::vector<cv::Point>& square, const int nbIteration);
	
}

#endif