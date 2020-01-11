//
// Created by 宋孝成 on 2019-04-30.
//

#ifndef DTAM_FRAME_HPP
#define DTAM_FRAME_HPP

#include <memory>
#include "Camera.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Core>
#include "utils/MatrixType.hpp"
#include <iostream>

namespace dtam {

class Frame {

 public:

  using Ptr = std::shared_ptr<Frame>;

  static Frame::Ptr create(
      double timeStamp,
      const cv::Mat &imageRGBUint8) {
    return std::make_shared<Frame>(timeStamp, imageRGBUint8);
  }

  Frame(double timeStamp,
        const cv::Mat &imageRGBUint8)
      : timeStamp_(timeStamp),
        imageRGBUint8_(imageRGBUint8),
        imageGrayUint8_(rgb2gray(imageRGBUint8)),
        imageRGBFloat_(uint82float(imageRGBUint8)),
        imageGrayFloat_(uint82float(imageGrayUint8_)) {}

  double getTimestamp() const { return timeStamp_; }

  const cv::Mat &getImageRGBUint8() const { return imageRGBUint8_; }

  const cv::Mat &getImageGrayUint8() const { return imageGrayUint8_; }

  const cv::Mat &getImageRGBFloat() const { return imageRGBFloat_; }

  const cv::Mat &getImageGrayFloat() const { return imageGrayFloat_; }

  const Vector3f getRGBFloat(float u, float v, int width, int height) const {
    int u0 = static_cast<int>(floorf(u));
    int u1;
    if (u0 == width - 1) {
      u1 = u0;
    } else {
      u1 = u0 + 1;
    }

    int v0 = static_cast<int>(floorf(v));
    int v1;
    if (v0 == height - 1) {
      v1 = v0;
    } else {
      v1 = v0 + 1;
    }

    const auto& I00 = imageRGBFloat_.at<cv::Vec3f>(v0, u0);
    const auto& I01 = imageRGBFloat_.at<cv::Vec3f>(v0, u1);
    const auto& I10 = imageRGBFloat_.at<cv::Vec3f>(v1, u0);
    const auto& I11 = imageRGBFloat_.at<cv::Vec3f>(v1, u1);

    float w00 = (v1-v)*(u1-u);
    float w01 = (v1-v)*(u-u0);
    float w10 = (v-v0)*(u1-u);
    float w11 = (v-v0)*(u-u0);

    const auto& bgr = w00*I00 + w01*I01 + w10*I10 + w11*I11;
    return Vector3f(bgr[2], bgr[1], bgr[0]);
  }

  const Vector3f getRGBFloat(int u, int v) const {
    const auto &bgr = imageRGBFloat_.at<cv::Vec3f>(v, u);
    return Vector3f(bgr[2], bgr[1], bgr[0]);
  }

  float getGrayFloat(float u, float v) const {
    return imageGrayFloat_.at<float>(static_cast<int>(lrint(v)),
                                static_cast<int>(lrint(u)));
  }

  float getGrayFloat(int u, int v) const {
    return imageGrayFloat_.at<float>(v, u);
  }

  int getWidth() const {
    return imageRGBUint8_.size().width;
  }

  int getHeight() const {
    return imageRGBUint8_.size().height;
  }


  const Matrix44f &getTcw() const { return tcw_; }

  const Matrix44f &getTwc() const { return twc_; }

  void setFeature(
      const std::vector<cv::KeyPoint> &keyPoints,
      const cv::Mat &descriptors) {
    keyPoints_ = keyPoints;
    descriptors_ = descriptors;
  }

  const std::vector<cv::KeyPoint> &getKeyPoints() const { return keyPoints_; }

  const cv::Mat &getDescriptor() const { return descriptors_; }

  void setTcw(const Matrix44f &tcw);

  void setTwc(const Matrix44f &twc);

 protected:

  static cv::Mat rgb2gray(const cv::Mat &src) {
    assert(src.type()==16); //CV_8UC3
    cv::Mat dest;
    cv::cvtColor(src, dest, cv::COLOR_RGB2GRAY);
    return dest;
  }

  static cv::Mat uint82float(const cv::Mat &src) {
    assert(src.depth()==0); //CV_8U
    cv::Mat dest;
    src.convertTo(dest, CV_32F, 1.0 / 255);
    return dest;
  }

  const double timeStamp_;

  // Order sensitive begin. According to c++ standard:
  // ...Then, non-static data members are initialized in order of declaration
  // in the class definition...
  const cv::Mat imageRGBUint8_;
  const cv::Mat imageGrayUint8_;
  const cv::Mat imageRGBFloat_;
  const cv::Mat imageGrayFloat_;
  // Order sensitive end.

  Matrix44f tcw_;
  Matrix44f twc_;
  std::vector<cv::KeyPoint> keyPoints_;
  cv::Mat descriptors_;
};

}

#endif //DTAM_FRAME_HPP
