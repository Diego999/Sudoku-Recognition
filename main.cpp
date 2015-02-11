#include "ImageProcessing/Controller/recognitioncontroller.h"
#include "Training/Controller/trainingcontroller.h"
#include "Training/Picture/picturecontroller.h"

#include <iostream>
#include <vector>

int findIdxMax(const std::vector<double>& v)
{
	double m = 0;
	int id = 0;

	for(int i = 0; i < v.size(); ++i)
	{
		if(v[i] > m)
		{
			m = v[i];
			id = i;
		}
		std::cout << "(" << (i+1) << "," << v[i] << ") ";
	}

	return id;
}

int main(int argc, char* argv[])
{
	RecognitionController recognition;
	TrainingController training;

	std::vector<std::pair<int, cv::Mat>> unrecognizedDigits = recognition.getDigitSudokuWebcam();
	std::vector<std::vector<double>> digitBlocks;

	for(auto& p : unrecognizedDigits)
		digitBlocks.push_back(PictureController::loadPictures(p.second));
	
	training.importANN("Training/newANN.ann");
	std::vector<std::vector<double>> recognitionResults = training.feed(digitBlocks);
	for(int i = 0; i < unrecognizedDigits.size(); ++i)
		std::cout << unrecognizedDigits[i].first << " : " << (1+findIdxMax(recognitionResults[i])) << " <- final choice" << std::endl;
	int i;
	std::cin >> i;
	return 0;
}