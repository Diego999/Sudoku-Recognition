#include "neuron.h"
#include "../Utils/utils.h"

#include <cassert>
#include <cmath>
#include <iostream>

const double Neuron::MAXIMUM_WEIGHT = 1.0;
const double Neuron::MINIMUM_WEIGHT = -1.0;

double Neuron::squashingFunction(double x)
{
    return 1.0/(1.0+exp(-x));
}

Neuron::Neuron(int nbInputs):m_nbInputs(nbInputs), m_delta(0), m_output(0), m_weights(nbInputs+1), m_prevDeltaWeights(nbInputs+1)
{
    assert(nbInputs > 0);
    initializeWeights();
    initializePrevDeltaWeights();
}

Neuron::~Neuron()
{

}

double Neuron::computeOutput(const std::vector<double>& inputs)
{
    double sum = 0.0;
    for(size_t i = 0; i < inputs.size(); ++i)
        sum += m_weights[i+1]*inputs[i]; // i+1 -> avoid threshold
    sum += m_weights[0]; //xi = 1.0

    m_output = squashingFunction(sum);
    return m_output;
}

void Neuron::initializeWeights()
{
    m_weights.clear();
    for(auto& val : ann_random::randDoubleList(MINIMUM_WEIGHT, MAXIMUM_WEIGHT, m_nbInputs+1))
        m_weights.push_back(val);
}

void Neuron::initializePrevDeltaWeights()
{
    m_prevDeltaWeights.clear();
    for(size_t i = 0; i < m_weights.size(); ++i)
        m_prevDeltaWeights.push_back(0);
}
