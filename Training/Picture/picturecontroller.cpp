#include "picturecontroller.h"

#include <opencv2/highgui/highgui.hpp>

std::vector<std::vector<double>> PictureController::loadPictures(const std::vector<std::string>& filepaths)
{
    std::vector<std::vector<double>> output;

    for(auto& filepath : filepaths)
        output.push_back(PictureController::getValuePixels(cv::imread(filepath, CV_LOAD_IMAGE_GRAYSCALE) > 128));
    
    return output;
}

std::vector<double> PictureController::loadPictures(const cv::Mat& image)
{
    return PictureController::getValuePixels(image > 128);
}

std::vector<double> PictureController::getValuePixels(const cv::Mat& picture)
{
    std::vector<double> output;
    for(int j = 0; j < picture.rows; ++j)
        for(int i = 0; i < picture.cols; ++i)
            output.push_back(!(static_cast<int>(picture.at<uchar>(j, i))/255));//Normalized form
    return output;
}