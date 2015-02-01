#include "artificialneuralnetwork.h"
#include "neuronlayer.h"
#include "neuron.h"

#include <cmath>
#include <cassert>

ArtificialNeuralNetwork::ArtificialNeuralNetwork(int nbInputs, int nbOutputs, const std::vector<int>& nbNeuronsPerHiddenLayer, double learningRate, double momentum)
    :m_nbInputs(nbInputs), m_nbOutputs(nbOutputs), m_nbNeuronsPerHiddenLayer(nbNeuronsPerHiddenLayer), m_learningRate(learningRate), m_momentum(momentum)
{
    // We don't treat the case of single perceptron.
    assert(nbNeuronsPerHiddenLayer.size() > 0);

    //First hidden layer
    m_layers.push_back(std::make_shared<NeuronLayer>(*m_nbNeuronsPerHiddenLayer.begin(), m_nbInputs));
    m_inputs.push_back(std::vector<double>(m_nbInputs));

    for(auto it = m_nbNeuronsPerHiddenLayer.begin()+1; it != m_nbNeuronsPerHiddenLayer.end(); ++it)
    {
        m_layers.push_back(std::make_shared<NeuronLayer>(*it, *(it-1)));
        m_inputs.push_back(std::vector<double>(*(it-1), 0));
    }

    m_inputs.push_back(std::vector<double>(*(m_nbNeuronsPerHiddenLayer.end()-1), 0));

    // Output layer. Output is considered as a layer, contrary to the inputs
    m_layers.push_back(std::make_shared<NeuronLayer>(m_nbOutputs, *(m_nbNeuronsPerHiddenLayer.end()-1)));
    m_inputs.push_back(std::vector<double>(m_nbOutputs, 0)); //To simplify the feedforward, the output of the ith-1 layer are the inputs of the ith layer. For the last layer (outputs), it is only outputs

    for(int i = 0; i < m_nbOutputs; ++i)
        m_targets.push_back(0);
}

ArtificialNeuralNetwork::ArtificialNeuralNetwork(const ArtificialNeuralNetwork& ann)
    :ArtificialNeuralNetwork(ann.m_nbInputs, ann.m_nbOutputs, ann.m_nbNeuronsPerHiddenLayer, ann.m_learningRate, ann.m_momentum)
{
    for(size_t i = 0; i < m_layers.size(); ++i)
        for(size_t j = 0; j < m_layers[i]->neurons().size(); ++j)
        {
            m_layers[i]->neuron(j)->threshold(ann.m_layers[i]->neuron(j)->threshold());
            for(int k = 0; k < m_layers[i]->neuron(j)->inputNb(); ++k)
                m_layers[i]->neuron(j)->weight(k, ann.m_layers[i]->neuron(j)->weight(k));
        }
}

ArtificialNeuralNetwork::~ArtificialNeuralNetwork()
{

}

const std::vector<double>& ArtificialNeuralNetwork::feedForward(const std::vector<double>& dataInputs)
{
    m_inputs[0] = dataInputs;
    layerForward();
    return m_inputs.back();
}

double ArtificialNeuralNetwork::train(const std::vector<double>& dataInputs, const std::vector<double>& dataTargets)
{
    m_inputs[0] = dataInputs;
    m_targets = dataTargets;

    layerForward();
    double errorTot = 0;
    errorTot += computeOutputError();
    errorTot += computeHiddenError();
    adjustWeights();
    return errorTot;
}

double ArtificialNeuralNetwork::validate(const std::vector<double>& dataInputs, const std::vector<double>& dataTargets)
{
    m_inputs[0] = dataInputs;
    m_targets = dataTargets;

    layerForward();
    double errorTot = 0;
    errorTot += computeOutputError();
    errorTot += computeHiddenError();
    return errorTot;
}

void ArtificialNeuralNetwork::layerForward()
{
    for(size_t i = 0; i < m_layers.size(); ++i)
        for(size_t j = 0; j < m_layers[i]->size(); ++j)
            m_inputs[i+1][j] = m_layers[i]->neuron(j)->computeOutput(m_inputs[i]);
}

double ArtificialNeuralNetwork::computeOutputError()
{
    double totErr = 0.0;
    for(size_t i = 0; i < m_layers.back()->size(); ++i)
        for(int j = 0; j < m_nbOutputs; ++j)
        {
            double o = m_inputs.back()[j];
            double t = m_targets[j];
            double delta = o*(1.0-o)*(t-o);
            m_layers.back()->deltaNeuron(j, delta);
            totErr += fabs(delta);
        }
    return totErr;
}

double ArtificialNeuralNetwork::computeHiddenError()
{
    double totErrAllHiddenLayer = 0.0;
    for(size_t i = m_layers.size()-1; i > 0 ; --i)
        for(size_t j = 0; j < m_layers[i-1]->size(); ++j)
        {
            double sum = 0.0;
            for(size_t k = 0; k < m_layers[i]->size(); ++k)
                sum += m_layers[i]->deltaNeuron(k)*m_layers[i]->weightNeuron(k, j);

            double h = m_layers[i-1]->outputNeuron(j);
            double delta = h*(1.0-h)*sum;
            m_layers[i-1]->deltaNeuron(j, delta);
            totErrAllHiddenLayer += fabs(delta);
        }
    return totErrAllHiddenLayer;
}

void ArtificialNeuralNetwork::adjustWeights()
{
    for(size_t i = 0; i < m_layers.size(); ++i)
        for(auto& neuron : m_layers[i]->neurons())
        {
            for(size_t j = 0; j < m_inputs[i].size(); ++j)
                neuron->updateWeight(j, m_learningRate*neuron->delta()*m_inputs[i][j] + m_momentum*neuron->prevDeltaWeight(j));
            neuron->updateThreshold(m_learningRate*neuron->delta() + m_momentum*neuron->prevDeltaThreshold());
        }
}

double ArtificialNeuralNetwork::threshold(int numLayer, int numNeuron) const
{
    return m_layers[numLayer]->threshold(numNeuron);
}

double ArtificialNeuralNetwork::deltaNeuron(int numLayer, int numNeuron) const
{
    return m_layers[numLayer]->deltaNeuron(numNeuron);
}

std::vector<double> ArtificialNeuralNetwork::weights(int numLayer, int numNeuron) const
{
    return m_layers[numLayer]->neuron(numNeuron)->weights();
}

std::vector<double> ArtificialNeuralNetwork::prevDeltaWeights(int numLayer, int numNeuron) const
{
    return m_layers[numLayer]->neuron(numNeuron)->prevDeltaWeights();
}

double ArtificialNeuralNetwork::weight(int numLayer, int numNeuron, int i) const
{
    return m_layers[numLayer]->neuron(numNeuron)->weight(i);
}

double ArtificialNeuralNetwork::prevDeltaWeight(int numLayer, int numNeuron,int i) const
{
    return m_layers[numLayer]->neuron(numNeuron)->prevDeltaWeight(i);
}

double ArtificialNeuralNetwork::prevDeltaThreshold(int numLayer, int numNeuron) const
{
    return m_layers[numLayer]->neuron(numNeuron)->prevDeltaThreshold();
}

void ArtificialNeuralNetwork::threshold(int numLayer, int numNeuron, double value)
{
    m_layers[numLayer]->neuron(numNeuron)->threshold(value);
}

void ArtificialNeuralNetwork::deltaNeuron(int numLayer, int numNeuron, double value)
{
    m_layers[numLayer]->neuron(numNeuron)->delta(value);
}
void ArtificialNeuralNetwork::weight(int numLayer, int numNeuron, int i, double weight)
{
    m_layers[numLayer]->neuron(numNeuron)->weight(i, weight);
}

void ArtificialNeuralNetwork::prevDeltaWeight(int numLayer, int numNeuron, int i, double v)
{
    return m_layers[numLayer]->neuron(numNeuron)->prevDeltaWeight(i, v);
}

void ArtificialNeuralNetwork::prevDeltaThreshold(int numLayer, int numNeuron, double d)
{
    return m_layers[numLayer]->neuron(numNeuron)->prevDeltaThreshold(d);
}
