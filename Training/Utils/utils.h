#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <limits>
#include <cmath>
#include <cassert>

namespace ann_random
{
    int randInt(int a, int b); // return a <= x <= b
    double randDouble(double a, double b);// return a <= x < b

    std::vector<int> randIntList(int a, int b, int n); // return a <= x <= b, n times
    std::vector<double> randDoubleList(double a, double b, int n);// return a <= x < b, n times
}

namespace ann_utils
{
    template<typename T>
    bool compare(const T a, const T b, T e = std::numeric_limits<T>::epsilon())
    {
        return fabs(a-b) < e;
    }

    template<typename T>
    void mergeVectors(std::vector<T>& v1, const std::vector<std::vector<T>>& v2)
    {
        for(auto& v:v2)
            v1.insert(v1.end(), v.begin(), v.end());
    }

    template<typename T>
    std::vector<std::vector<T>> createSubSamples(const std::vector<T>& sets, const unsigned int k)
    {
        assert(k >= 0);
        std::vector<std::vector<T>> samples;
        int sizeSample = sets.size()/k;
        for(unsigned int i = 0; i*sizeSample < sets.size(); ++i)
            if(sets.begin()+(i+1)*sizeSample > sets.end())
                samples.back().insert(samples.back().end(), sets.begin()+i*sizeSample, sets.end());
            else
            {
                samples.push_back(std::vector<T>());
                samples.back().insert(samples.back().end(), sets.begin()+i*sizeSample, sets.begin()+(i+1)*sizeSample);
            }
        return samples;
    }
}

#endif // UTILS_H
