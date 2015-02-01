#include "neuronlayer.h"
#include "neuron.h"

#include <cassert>

NeuronLayer::NeuronLayer(int nbNeurons, int nbInputs):m_neurons(nbNeurons)
{
    assert(nbNeurons > 0);
    for(int i = 0; i < nbNeurons; ++i)
        m_neurons[i] = std::make_shared<Neuron>(nbInputs);
}

NeuronLayer::~NeuronLayer()
{

}

void NeuronLayer::updateWeightNeuron(int neuron, int i, double v)
{
    return m_neurons[neuron]->updateWeight(i, v);
}

void NeuronLayer::weightNeuron(int neuron, int i, double v)
{
    m_neurons[neuron]->weight(i, v);
}

void NeuronLayer::deltaNeuron(int neuron, double d)
{
    m_neurons[neuron]->delta(d);
}

void NeuronLayer::threshold(int neuron, double v)
{
    m_neurons[neuron]->threshold(v);
}

double NeuronLayer::outputNeuron(int neuron) const
{
    return m_neurons[neuron]->output();
}

double NeuronLayer::weightNeuron(int neuron, int i) const
{
    return m_neurons[neuron]->weight(i);
}

double NeuronLayer::deltaNeuron(int neuron) const
{
    return m_neurons[neuron]->delta();
}

double NeuronLayer::threshold(int neuron) const
{
    return m_neurons[neuron]->threshold();
}
