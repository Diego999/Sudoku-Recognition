#include "recognitioncontroller.h"
#include "../Utils/imageprocessingutils.h"

#include <iostream>

RecognitionController::RecognitionController()
{

}

RecognitionController::~RecognitionController()
{

}

void RecognitionController::captureSudoku(const double minAreaPercentageFilter, const double maxAreaPercentageFilter, const int nbIteration)
{
	imgSudokuStream = image_processing_utils::captureSoduku(minAreaPercentageFilter, maxAreaPercentageFilter, exteriorSquareSudoku,nbIteration);
}

void RecognitionController::cropSudoku(const int destW, const int destH)
{
	imgSudokuCropped = image_processing_utils::cropPicture(exteriorSquareSudoku, imgSudokuStream, destW, destH);
}

void RecognitionController::findAndLabelBlocks(const double minAreaFactor, const double maxAreaFactor, const int finalWBlock, const int finalHBlock)
{
    double area = image_processing_utils::computeAreaContour(exteriorSquareSudoku);
    double areaOneBlock = (area/(image_processing_utils::SUDOKU_SIZE*image_processing_utils::SUDOKU_SIZE));

	double minAreaPercentageFilter = 100.0*minAreaFactor*areaOneBlock/area;
    double maxAreaPercentageFilter = 100.0*maxAreaFactor*areaOneBlock/area;

    std::vector<std::vector<cv::Point>> uniqueSquares = image_processing_utils::findSquares(imgSudokuCropped, minAreaPercentageFilter, maxAreaPercentageFilter, area);
    if(!image_processing_utils::removeDuplicateSquares(uniqueSquares))
    {
    	std::cerr << "Some blocks haven't been detected. Please try again with better conditions" << std::endl;
    	exit(-1);
    }
    else
    {
    	imgSudokuSortedBlocks = image_processing_utils::extractBlocks(uniqueSquares, imgSudokuCropped);
    	imgSudokuLabelledDigits = image_processing_utils::extractDigitBlocks(imgSudokuSortedBlocks, finalWBlock, finalHBlock);
    }
}

std::vector<std::pair<int, cv::Mat>> RecognitionController::getDigitSudokuWebcam()
{
	captureSudoku();
	cropSudoku();
	findAndLabelBlocks();

	return imgSudokuLabelledDigits;
}
