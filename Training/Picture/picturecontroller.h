#ifndef PICTURECONTROLLER_H
#define PICTURECONTROLLER_H

#include <opencv2/highgui/highgui.hpp>

#include <vector>

class PictureController
{
public:
	static std::vector<std::vector<double>> loadPictures(const std::vector<std::string>& filepaths);
    static std::vector<double> loadPictures(const cv::Mat& img);

private:
	static std::vector<double> getValuePixels(const cv::Mat& picture);
};
#endif // PICTURECONTROLLER_H
