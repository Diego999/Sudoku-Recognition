#include "utils.h"

#include <random>
#include <vector>
#include <ctime>

int ann_random::randInt(int a, int b)
{
    return randIntList(a, b, 1)[0];
}

double ann_random::randDouble(double a, double b)
{
    return randDoubleList(a, b, 1)[0];
}

std::vector<int> ann_random::randIntList(int a, int b, int n)
{
    static std::default_random_engine generator(time(0));
    std::uniform_int_distribution<int> distribution(a, b);
    std::vector<int> output(n);

    for(int i = 0; i < n; ++i)
            output[i] = distribution(generator);

    return output;
}

std::vector<double> ann_random::randDoubleList(double a, double b, int n)
{
    static std::default_random_engine generator(time(0));
    std::uniform_real_distribution<double> distribution(a, b);
    std::vector<double> output(n);

    for(int i = 0; i < n; ++i)
        output[i] = distribution(generator);

    return output;
}
