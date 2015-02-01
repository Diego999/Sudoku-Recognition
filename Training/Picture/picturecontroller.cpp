#include "picturecontroller.h"

#include <opencv2/highgui/highgui.hpp>

std::vector<std::vector<double>> PictureController::loadPictures(const std::vector<std::string>& filepaths)
{
    std::vector<std::vector<double>> output;

    for(auto& filepath : filepaths)
    {
        cv::Mat picture = cv::imread(filepath, CV_LOAD_IMAGE_GRAYSCALE) > 128;;

        std::vector<double> data;
        for(int j = 0; j < picture.rows; ++j)
            for(int i = 0; i < picture.cols; ++i)
                data.push_back(static_cast<int>(picture.at<uchar>(j, i))/255); //Normalized form
        output.push_back(data);
    }
    return output;
}
