#ifndef PICTURECONTROLLER_H
#define PICTURECONTROLLER_H

#include <vector>
#include <string>

class PictureController
{
public:
    static std::vector<std::vector<double>> loadPictures(const std::vector<std::string>& filepaths);
};

#endif // PICTURECONTROLLER_H
