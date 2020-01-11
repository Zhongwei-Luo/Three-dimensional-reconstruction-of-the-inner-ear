//
// Created by 宋孝成 on 2019-05-02.
//

#ifndef DTAM_MATRIXTYPE_HPP
#define DTAM_MATRIXTYPE_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
namespace dtam {

using Matrix34f = Eigen::Matrix<float, 3, 4>;
using Matrix44f = Eigen::Matrix4f;
using Matrix33f = Eigen::Matrix3f;
using VectorXf = Eigen::VectorXf;
using Vector4f = Eigen::Vector4f;
using Vector3f = Eigen::Vector3f;
using Vector2f = Eigen::Vector2f;
using Vector3i = Eigen::Vector3i;
using Matrix4Xf = Eigen::Matrix4Xf;
using Matrix3Xf = Eigen::Matrix3Xf;
using Matrix2Xf = Eigen::Matrix2Xf;
using MatrixXXf = Eigen::MatrixXf;
using VectorXc = Eigen::VectorXcf;
using VectorXb = Eigen::Matrix<bool, Eigen::Dynamic, 1>;
using Quaternionf = Eigen::Quaternionf;

struct cv_T
{
    cv::Mat mT;cv::Mat mR;cv::Mat mt;
};
}

#endif //DTAM_MATRIXTYPE_HPP
