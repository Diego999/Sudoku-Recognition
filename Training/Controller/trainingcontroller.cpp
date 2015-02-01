#include "TrainingController.h"
#include "../Picture/PictureController.h"

#include <iostream>
#include <fstream>
#include <sstream>

TrainingController::TrainingController():
    m_annController(nullptr)
{
}

TrainingController::~TrainingController()
{
    if(m_annController != nullptr)
        m_annController->stopTraining();
}

double TrainingController::testValidity(const std::vector<std::string>& filepaths, const char separator) const
{
    std::vector<std::vector<double>> inputSet = PictureController::loadPictures(filepaths);
    std::vector<std::vector<double>> targetSet = TrainingController::generateTargets(filepaths, separator);

    std::vector<std::pair<std::vector<double>, std::vector<double>>> set;

    for(size_t i = 0; i < inputSet.size(); ++i)
        set.push_back({inputSet[i], targetSet[i]});
    return m_annController->test(set);
}

const std::vector<std::vector<double>> TrainingController::generateTargets(const std::vector<std::string>& filepaths, const char separator)
{
    std::vector<std::vector<double>> targets;
    for(auto& file : filepaths)
    {
        std::vector<double> target;
        std::vector<std::string> tokens;
        std::istringstream f(file);
        std::string s;
        while(std::getline(f, s, separator))
            tokens.push_back(s);
        
        int digit = tokens[tokens.size()-2].back()-'0';
        for(int i = 1; i <= 9; ++i)
            target.push_back(digit == i ? 0.9 : 0.1);
        targets.push_back(target);
    }
    return targets;
}

void TrainingController::importANN(const std::string& annPath)
{
    m_annController = std::make_shared<ANNController>(annPath);
}

void TrainingController::exportANN(const std::string& annPath)
{
    m_annController->exportANN(annPath);
}

void TrainingController::configANN(const std::vector<int>& nbNeuronsPerHiddenLayer, double learningRate, double momentum, double error, bool kFoldCrossValidation, unsigned int k)
{
     m_nbNeuronsPerHiddenLayer = nbNeuronsPerHiddenLayer;
     m_learningRate = learningRate;
     m_momentum = momentum;
     m_error = error;
     m_kFoldCrossValidation = kFoldCrossValidation;
     m_k = k;
}

int TrainingController::readSetPath(const std::string& setPath, std::vector<std::pair<std::vector<double>, std::vector<double>>>& set, const char separator)
{
    std::vector<std::string> currentFilePaths;
    std::ifstream file(setPath);
    if(!file.is_open())
    {
        std::cerr << "[Can't open file] -> " << setPath << std::endl;
        return 0;
    }

    std::string in;
    while (getline(file,in))
        currentFilePaths.push_back(in);
    file.close();

    std::vector<std::vector<double>> inputsSet = PictureController::loadPictures(currentFilePaths);
    std::vector<std::vector<double>> targetsSet = generateTargets(currentFilePaths, separator);

    set.clear();
    for(size_t i = 0; i < inputsSet.size(); ++i)
        set.push_back({inputsSet[i], targetsSet[i]});

    return set.size();
}

int TrainingController::setTrainingSetPath(const std::string& trainingSetPath, const char separator)
{
    return TrainingController::readSetPath(trainingSetPath, m_trainingSet, separator);
}

int TrainingController::setValidationSetPath(const std::string& validationSetPath, const char separator)
{
    return TrainingController::readSetPath(validationSetPath, m_validationSet, separator);
}

void TrainingController::startTraining()
{
    if(!m_trainingSet.empty())
    {
        if(!m_validationSet.empty())
            m_annController = std::make_shared<ANNController>(m_nbNeuronsPerHiddenLayer, m_learningRate, m_momentum, m_trainingSet, m_validationSet);
        else
            m_annController = std::make_shared<ANNController>(m_nbNeuronsPerHiddenLayer, m_learningRate, m_momentum, m_trainingSet);
    }

    m_annController->error(m_error);

    std::function<void(void)> didFinish = [&](void) { }; //No GUI

    std::cout << "[Training start]" << std::endl;
    if(m_kFoldCrossValidation)
    {
        std::function<void(long, std::vector<double>&, std::vector<double>&)> callback = [&](long i, std::vector<double> &errT, std::vector<double>& errV){ }; //No GUI
        std::function<void(long, double)> callbackFinalANN = [&](long i, double err) {}; //No GUI
        
        m_annController->kFoldCrossValidation(callback, callbackFinalANN, m_k, didFinish);
    }
    else
    {
        std::function<void(long, double, double)> callback = [&](long iteration, double trainingError, double testingError){ 
            static int i = 0;
            if(++i%500 == 0)
                std::cout << iteration << " " << trainingError << " " << testingError << std::endl;
        }; //No GUI

        m_annController->train(callback, didFinish);
    }
}

std::vector<std::vector<double>> TrainingController::feed(const std::vector<std::string>& filepaths)
{
    std::vector<std::vector<double>> pictures = PictureController::loadPictures(filepaths);
    std::vector<std::vector<double>> fullResults = m_annController->feedForward(pictures);
    std::vector<std::vector<double>> results;
    for(auto& result : fullResults)
        results.push_back(result);
    return results;
}

void TrainingController::reset()
{
    if(m_annController == nullptr) return;
    m_annController->stopTraining();
    m_annController.reset();
    m_trainingSet.clear();
    m_validationSet.clear();
}

void TrainingController::stopTraining()
{
    m_annController->stopTraining();
}
