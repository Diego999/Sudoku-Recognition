//#include "Controller/trainingcontroller.h"
//#include <iostream>
//#include <string>
//#include <fstream>
//
//int main(int argc, char* argv[])
//{
//	const char SEPARATOR = '_';
//	const std::string TRAINING_LIST = "/Users/Diego/Github/Sudoku-Recognition/Training/Sets/training.list";
//	const std::string VALIDATION_LIST = "/Users/Diego/Github/Sudoku-Recognition/Training/Sets/validation.list";
//	TrainingController controller;
//
//	std::vector<int> nbNeuronsPerHiddenLayer({20});
//	double learningRate = 0.3;
//	double momentum = 0.9;
//	double error = 0.1;
//	bool kFoldCrossValidation = false;
//	unsigned int k = 1;
//	controller.configANN(nbNeuronsPerHiddenLayer, learningRate, momentum, error, kFoldCrossValidation, k);
//	controller.setTrainingSetPath(TRAINING_LIST, SEPARATOR);
//	controller.setValidationSetPath(VALIDATION_LIST, SEPARATOR);
//	controller.startTraining();
//	controller.exportANN("newANN.ann");
//
//	//Test validity
//	/*controller.importANN("newANN.ann");
//
//	std::vector<std::string> currentFilePaths;
//    std::ifstream file("/Users/Diego/Github/Sudoku-Recognition/Training/Sets/all_training.list");
//    std::string in;
//    while (getline(file,in))
//        currentFilePaths.push_back(in);
//    file.close();
//
//    std::cout << controller.testValidity(currentFilePaths, SEPARATOR) << std::endl;
//	*/
//	return 0;
//}//