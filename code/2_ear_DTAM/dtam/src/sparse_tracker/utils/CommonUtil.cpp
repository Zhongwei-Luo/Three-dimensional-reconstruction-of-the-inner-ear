//
// Created by songxiaocheng on 19-5-10.
//

#include "ScalePropagationUtil.hpp"
#include "CommonUtil.hpp"
#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <random>

using namespace std;

namespace dtam {

std::vector<dtam::Vector3f> relativeTriangulate(
    const std::vector<cv::Point2d> &points1,
    const std::vector<cv::Point2d> &points2,
    const dtam::Matrix33f &K,
    const dtam::Matrix44f &T21) {
  cv::Mat cP1 = cv::Mat::eye(3, 4, CV_32F);
  cv::Mat cP2;
  eigen2cv(T21.topRows(3).eval(), cP2);
  std::vector<cv::Point2d> pts_1, pts_2;
  const float fx = K(0, 0);
  const float fy = K(1, 1);
  const float cx = K(0, 2);
  const float cy = K(1, 2);
  for (std::size_t i = 0; i < points1.size(); i++) {
    const auto &p1 = points1[i];
    const auto &p2 = points2[i];
    pts_1.push_back(cv::Point2f((p1.x - cx) / fx, (p1.y - cy) / fy));
    pts_2.push_back(cv::Point2f((p2.x - cx) / fx, (p2.y - cy) / fy));
  }
  cv::Mat p4d;
  triangulatePoints(cP1, cP2, pts_1, pts_2, p4d);
  std::vector<dtam::Vector3f> p3d;
  for (int i = 0; i < p4d.cols; i++) {
    const dtam::Vector3f p(
        p4d.at<double>(0, i) / p4d.at<double>(3, i),
        p4d.at<double>(1, i) / p4d.at<double>(3, i),
        p4d.at<double>(2, i) / p4d.at<double>(3, i)
    );
    p3d.emplace_back(p);
  }
  return p3d;
}

// for debug only
void drawMatches(
    const Frame::Ptr &frame1,
    const Frame::Ptr &frame2,
    const std::vector<cv::DMatch> &matches) {
  cv::Mat a = frame1->getImageRGBFloat().clone();
  std::vector<cv::Point2d> pixelPoints1, pixelPoints2;
  for (const auto match:matches) {
    const auto p1 = frame1->getKeyPoints()[match.queryIdx].pt;
    const auto p2 = frame2->getKeyPoints()[match.trainIdx].pt;
    pixelPoints1.emplace_back(p1);
    pixelPoints2.emplace_back(p2);
  }
  for (size_t i = 0; i < pixelPoints1.size(); i++) {
    cv::arrowedLine(a,
                    pixelPoints1[i],
                    pixelPoints2[i],
                    CV_RGB(255, 0, 0));
  }
  cv::imshow("test", a);
  cv::waitKey(0);
}

Matrix3Xf triangulatePoints(const Matrix2Xf &points1,
                            const Matrix2Xf &points2,
                            const Matrix34f &projectionMat1,
                            const Matrix34f &projectionMat2) {
  Matrix3Xf points3d;
  points3d.resize(3, points1.cols());
  Matrix44f A;
  for (int i = 0; i < points1.cols(); i++) {
    A << points1(0, i) * projectionMat1.row(2) - projectionMat1.row(0),
        points1(1, i) * projectionMat1.row(2) - projectionMat1.row(1),
        points2(0, i) * projectionMat2.row(2) - projectionMat2.row(0),
        points2(1, i) * projectionMat2.row(2) - projectionMat2.row(1);
    Eigen::JacobiSVD<Matrix44f> svd(A, Eigen::ComputeFullV);
    const auto &V = svd.matrixV();
    points3d.col(i) = V.col(3).head(3) / V(3, 3);
  }
  return points3d;
}

vector<cv::DMatch> randomSampleMatches(
    const vector<cv::DMatch> &matches,
    int nSample) {
  vector<cv::DMatch> sampledMatches;
  sample(matches.begin(),
         matches.end(),
         std::back_inserter(sampledMatches),
         nSample,
         std::mt19937{std::random_device{}()});
  return sampledMatches;
}

vector<cv::DMatch> selectMatches(
    const vector<cv::DMatch> &matches,
    const VectorXb &mask) {
  vector<cv::DMatch> selectedMatches;
  for (int i = 0; i < mask.size(); i++) {
    if (mask(i)) selectedMatches.emplace_back(matches[i]);
  }
  return selectedMatches;
}

tuple<Matrix2Xf, Matrix2Xf> getPixelPoints(
    const Frame::Ptr &frame1,
    const Frame::Ptr &frame2,
    const vector<cv::DMatch> &matches) {
  Matrix2Xf pixelPoints1, pixelPoints2;
  pixelPoints1.resize(Eigen::NoChange, matches.size());
  pixelPoints2.resize(Eigen::NoChange, matches.size());
  for (size_t i = 0; i < matches.size(); i++) {
    const auto p1 = frame1->getKeyPoints()[matches[i].queryIdx].pt;
    const auto p2 = frame2->getKeyPoints()[matches[i].trainIdx].pt;
    pixelPoints1.col(i) = Vector2f(p1.x, p1.y);
    pixelPoints2.col(i) = Vector2f(p2.x, p2.y);
  }
  return {pixelPoints1, pixelPoints2};
}

}