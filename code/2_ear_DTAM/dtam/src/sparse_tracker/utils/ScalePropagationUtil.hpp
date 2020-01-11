//
// Created by songxiaocheng on 19-5-9.
//

#ifndef DTAM_SCALEPROPAGATIONUTIL_HPP
#define DTAM_SCALEPROPAGATIONUTIL_HPP

#include <tuple>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Frame.hpp"

namespace dtam {

std::tuple<std::vector<cv::Point2d>,
           std::vector<cv::Point2d>,
           std::vector<cv::Point2d>>
findCommonPoints(
    const Frame::Ptr &frame1,
    const Frame::Ptr &frame2,
    const Frame::Ptr &frame3,
    const std::vector<cv::DMatch> &matches21,
    const std::vector<cv::DMatch> &matches23);

float relativeScale(
    const Frame::Ptr &frame1,
    const Frame::Ptr &frame2,
    const Frame::Ptr &frame3,
    const std::vector<cv::DMatch> &matches21,
    const std::vector<cv::DMatch> &matches23,
    const Matrix33f &cameraMat,
    const Matrix44f &T32);

}

#endif //DTAM_SCALEPROPAGATIONUTIL_HPP
