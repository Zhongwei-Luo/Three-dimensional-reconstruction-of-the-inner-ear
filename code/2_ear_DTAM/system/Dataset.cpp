//
// Created by 宋孝成 on 2019-05-04.
//

#include <iomanip>
#include <iostream>
#include <regex>
#include <fstream>
#include "Dataset.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "utils/MatrixType.hpp"

using namespace std;
using namespace std::experimental;

inline string Dataset::getMainFileName() const {
  ostringstream oStr;
  oStr<< setfill('0') << setw(6) << (index_ );
  return oStr.str();
}

void Dataset::reset() {
  index_ = 0;
}
/*
bool Dataset::next() {
  index_++;
  // The frame 326 is missing in the dataset
  if (index_ == 326) {
    return next();
  }
  return index_ <= maxNumber_
      && filesystem::exists(path_ / (getMainFileName() + ".png"))
      && filesystem::exists(path_ / (getMainFileName() + ".txt"));
}
 */
//fixbug 如果出现了什么问题，你就把上面的给恢复了
bool Dataset::next()
{
    index_++;
    // The frame 326 is missing in the dataset
    if(index_>200)
    {return false;}
    else{return true;}
}




bool Dataset::back() {
  index_--;
  // The frame 326 is missing in the dataset
  if (index_ == 326) {
    return back();
  }
  return index_ <= maxNumber_
      && filesystem::exists(path_ / (getMainFileName() + ".png"))
      && filesystem::exists(path_ / (getMainFileName() + ".txt"));
}

double Dataset::readTimestamp() const {
  return (index_ - 1) / 30.0f;
}

cv::Mat Dataset::readRGB() const {
  string filename = path_ / (getMainFileName() + ".png");
  cout<<"this is the name of readRGB : "<<filename<<endl;
  return cv::imread(filename);
}

Eigen::Matrix4f Dataset::readTwc(std:: vector<dtam::cv_T> &vec) const
{

Eigen::Matrix4f T;


cv2eigen(vec[index_].mT,T);
//cout<<"index_ : "<<index_<<endl;
//cout<<"Mat : "<<vec[index_].mT<<endl<<"eigen: "<<T<<endl;


  return T;
}