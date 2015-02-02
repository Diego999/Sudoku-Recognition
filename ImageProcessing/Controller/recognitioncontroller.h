#ifndef RECOGNITION_CONTROLLER_H
#define RECOGNITION_CONTROLLER_H

#include <opencv2/highgui/highgui.hpp>

#include <vector>

class RecognitionController
{
public:
	RecognitionController();
	~RecognitionController();

	void captureSudoku(const double minAreaPercentageFilter = 20, const double maxAreaPercentageFilter = 50, const int nbIteration = 10);
	void cropSudoku(const int destW = 640, const int destH = 480);
	void findAndLabelBlocks(const double minAreaFactor = 1.5, const double maxAreaFactor = 2.5);
	

private:
	std::vector<cv::Point> exteriorSquareSudoku;

	cv::Mat imgSudokuStream;
	cv::Mat imgSudokuCropped;

	std::vector<std::vector<cv::Point>> labelledSudokuBlocks;
	std::vector<cv::Mat> imgSudokuBlocks;
	std::vector<std::pair<int, cv::Mat>> imgSudokuLabelledDigits;
};

#endif