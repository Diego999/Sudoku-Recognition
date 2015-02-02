#include "recognitioncontroller.h"
#include "../Utils/imageprocessingutils.h"

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

void RecognitionController::findAndLabelBlocks(const double minAreaFactor, const double maxAreaFactor)
{
    double area = image_processing_utils::computeAreaContour(exteriorSquareSudoku);
    double areaOneBlock = (area/(image_processing_utils::SUDOKU_SIZE*image_processing_utils::SUDOKU_SIZE));

	double minAreaPercentageFilter = 100.0*minAreaFactor*areaOneBlock/area;
    double maxAreaPercentageFilter = 100.0*maxAreaFactor*areaOneBlock/area;

    //labelledSudokuBlocks = ;
    labelledSudokuBlocks = image_processing_utils::labelling(image_processing_utils::findSquares(imgSudokuCropped, minAreaPercentageFilter, maxAreaPercentageFilter, area));
}

int main(int argc, char* argv[])
{
	RecognitionController controller;
	controller.captureSudoku();
	controller.cropSudoku();
	//controller.findAndLabelBlocks();
	return 0;
}