#ifndef NEURON_H
#define NEURON_H

#include <vector>

class Neuron
{
public:
    Neuron(int nbInputs);
    ~Neuron();

    double computeOutput(const std::vector<double>& inputs);

    void updateWeight(int i, double v) {m_prevDeltaWeights[i+1] = v; m_weights[i+1] += v;}
    void updateThreshold(double v) {m_prevDeltaWeights[0] = v; m_weights[0] += v;}

    std::vector<double> weights() const {return std::vector<double>(m_weights.begin()+1, m_weights.end());}
    std::vector<double> prevDeltaWeights() const {return std::vector<double>(m_prevDeltaWeights.begin()+1, m_prevDeltaWeights.end());}
    size_t size() const {return m_weights.size()-1;}
    int inputNb() const {return m_nbInputs;}
    double output() const {return m_output;}

    double threshold() const {return m_weights[0];}
    double weight(int i) const {return m_weights[i+1];}
    double prevDeltaWeight(int i) const {return m_prevDeltaWeights[i+1];}
    double prevDeltaThreshold() const {return m_prevDeltaWeights[0];}
    double delta() const {return m_delta;}

    void threshold(double t) {m_weights[0]=t;}
    void weight(int i, double v) {m_weights[i+1] = v;}
    void prevDeltaWeight(int i, double v) { m_prevDeltaWeights[i+1] = v;}
    void prevDeltaThreshold(double d) { m_prevDeltaWeights[0] = d;}
    void delta(double d) {m_delta=d;}

private:
    static const double MINIMUM_WEIGHT;
    static const double MAXIMUM_WEIGHT;

    static double squashingFunction(double x);

    void initializeWeights();
    void initializePrevDeltaWeights();

    int m_nbInputs;
    double m_delta;
    double m_output;

    std::vector<double> m_weights;
    std::vector<double> m_prevDeltaWeights; // For the momentum
};

#endif // NEURON_H
