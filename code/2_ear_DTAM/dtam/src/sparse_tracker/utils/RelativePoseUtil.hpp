//
// Created by songxiaocheng on 19-5-10.
//

#ifndef DTAM_RELATIVEPOSEUTIL_HPP
#define DTAM_RELATIVEPOSEUTIL_HPP

#include <tuple>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Frame.hpp"
#include <opencv2/core/eigen.hpp>

using namespace std;

namespace dtam {

std::tuple<Matrix33f, Vector3f, VectorXb> extractRelativePoseOpenCV(
    const Frame::Ptr &frame1,
    const Frame::Ptr &frame2,
    const std::vector<cv::DMatch> &matches,
    const Matrix33f &cameraMat);

std::tuple<Matrix33f, Vector3f, VectorXb> extractRelativePose(
    const Frame::Ptr &frame1,
    const Frame::Ptr &frame2,
    const std::vector<cv::DMatch> &matches,
    const Matrix33f &cameraMat);

}

#endif //DTAM_RELATIVEPOSEUTIL_HPP
