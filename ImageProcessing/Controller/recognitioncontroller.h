#ifndef RECOGNITION_CONTROLLER_H
#define RECOGNITION_CONTROLLER_H

#include <opencv2/highgui/highgui.hpp>

#include <vector>

class RecognitionController
{
public:
	RecognitionController();
	~RecognitionController();

	std::vector<std::pair<int, cv::Mat>> getDigitSudokuWebcam();

private:
	void captureSudoku(const double minAreaPercentageFilter = 20, const double maxAreaPercentageFilter = 50, const int nbIteration = 10);
	void cropSudoku(const int destW = 640, const int destH = 480);
	void findAndLabelBlocks(const double minAreaFactor = 1.5, const double maxAreaFactor = 2.5, const int finalWBlock = 60, const int finalHBlock = 40);

	std::vector<cv::Point> exteriorSquareSudoku;

	cv::Mat imgSudokuStream;
	cv::Mat imgSudokuCropped;

	std::vector<cv::Mat> imgSudokuSortedBlocks; //Sorted in row major
	std::vector<std::pair<int, cv::Mat>> imgSudokuLabelledDigits;
};

#endif