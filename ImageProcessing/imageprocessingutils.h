#ifndef IMAGE_PROCESSING_UTILS_H
#define IMAGE_PROCESSING_UTILS_H

#include <vector>
#include <string>
#include <opencv2/highgui/highgui.hpp>

namespace image_processing_utils
{
	double findAngle(cv::Point p1, cv::Point p2, cv::Point p0);
	std::vector<std::vector<cv::Point>> findSquares(const cv::Mat& image, double minAreaPercentageFilter, double maxAreaPercentageFilter, double wholeArea, const int thresh = 50, const int N = 11);
	void drawSquares(cv::Mat& image, const std::vector<std::vector<cv::Point>>& squares, bool R = false, bool G = false, bool B = false);
	cv::Mat captureSoduku(const int minAreaPercentageFilter, const int maxAreaPercentageFilter, std::vector<cv::Point>& square, const int nbIteration);
	void prepareQuadri(std::vector<cv::Point>& quadri);
	cv::Mat cropPicture(const std::vector<cv::Point>& srcQuadri, const cv::Mat& srcImg, const int w, const int h);
	bool removeDuplicateSquares(std::vector<std::vector<cv::Point>>& squares);
	std::vector<std::vector<cv::Point>> labelling(const std::vector<std::vector<cv::Point>>& squares);
	std::vector<cv::Mat> extractBlocks(const std::vector<std::vector<cv::Point>>& squares, const cv::Mat& image);
	

	static const std::string WEBCAM_WINDOW_TITLE = "Stream";
	static const std::string CROP_SUDOKU_WINDOW_TITLE = "Croped Sudoku";
}

#endif