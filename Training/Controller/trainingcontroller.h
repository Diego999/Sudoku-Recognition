#ifndef TRAININGCONTROLLER_H
#define TRAININGCONTROLLER_H

#include "../NeuralNetwork/anncontroller.h"
#include <memory>

class TrainingController
{
public:
    TrainingController();
    ~TrainingController();

    double testValidity(const std::vector<std::string>& filepaths, const char separator) const;

    void importANN(const std::string& annPath);
    void exportANN(const std::string& annPath);
    void configANN(const std::vector<int>& nbNeuronsPerHiddenLayer, double learningRate, double momentum, double error, bool kFoldCrossValidation, unsigned int k = 1);
    int setTrainingSetPath(const std::string& trainingSetPath, const char separator);
    int setValidationSetPath(const std::string& validationSetPath, const char separator);
    void startTraining();
    std::vector<std::vector<double>> feed(const std::vector<std::string>& filepaths);
    
    static const std::vector<std::vector<double>> generateTargets(const std::vector<std::string>& filepaths, const char separator);
    
    bool hasValidationSet() const { return !m_validationSet.empty(); }
    void reset();
    void stopTraining();

private:
    static int readSetPath(const std::string& setPath, std::vector<std::pair<std::vector<double>, std::vector<double>>> & set, const char separator);

    std::shared_ptr<ANNController> m_annController;
    std::vector<int> m_nbNeuronsPerHiddenLayer;
    double m_learningRate;
    double m_momentum;
    double m_error;
    bool m_kFoldCrossValidation;
    unsigned int m_k;
    std::vector<std::pair<std::vector<double>, std::vector<double>>> m_trainingSet;
    std::vector<std::pair<std::vector<double>, std::vector<double>>> m_validationSet;
};

#endif // IPPCONTROLLER_H
