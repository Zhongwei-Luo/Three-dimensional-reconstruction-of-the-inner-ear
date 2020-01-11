//
// Created by songxiaocheng on 19-5-10.
//

#ifndef DTAM_COMMONUTIL_HPP
#define DTAM_COMMONUTIL_HPP

namespace dtam {

std::vector<dtam::Vector3f> relativeTriangulate(
    const std::vector<cv::Point2d> &points1,
    const std::vector<cv::Point2d> &points2,
    const dtam::Matrix33f &K,
    const dtam::Matrix44f &T12);

inline Matrix2Xf camera2pixel(
    const Matrix2Xf &cameraPoints,
    const Matrix33f &cameraMat) {
  const auto c = Vector2f(cameraMat(0, 2), cameraMat(1, 2)).array();
  const auto f = Vector2f(cameraMat(0, 0), cameraMat(1, 1)).array();
  return ((cameraPoints.array().colwise() * f).colwise() + c).matrix().eval();
}

inline Matrix2Xf pixel2camera(
    const Matrix2Xf &pixelsPoints,
    const Matrix33f &cameraMat) {
  const auto c = Vector2f(cameraMat(0, 2), cameraMat(1, 2)).array();
  const auto f = Vector2f(cameraMat(0, 0), cameraMat(1, 1)).array();
  return ((pixelsPoints.array().colwise() - c).colwise() / f).matrix().eval();
}

inline Matrix2Xf normalizedPixels(
    const Matrix2Xf &pixelsPoints,
    int width,
    int height) {
  return ((pixelsPoints.array().colwise() / Vector2f(width, height).array()) * 2
      - 1).matrix().eval();
}

inline Matrix2Xf space2camera(
    const Matrix3Xf &spacePoints,
    const Matrix33f &rotation,
    const Vector3f &translation) {
  const auto p3d = (rotation * spacePoints).colwise() + translation;
  return p3d.colwise().hnormalized().topRows(2).eval();
}

inline Matrix2Xf space2pixel(
    const Matrix3Xf &spacePoints,
    const Matrix33f &cameraMat,
    const Matrix33f &rotation,
    const Vector3f &translation) {
  return camera2pixel(space2camera(spacePoints, rotation, translation),
                      cameraMat);
}

void drawMatches(
    const Frame::Ptr &frame1,
    const Frame::Ptr &frame2,
    const std::vector<cv::DMatch> &matches);

Matrix3Xf triangulatePoints(
    const Matrix2Xf &points1,
    const Matrix2Xf &points2,
    const Matrix34f &projectionMat1,
    const Matrix34f &projectionMat2);

std::vector<cv::DMatch> randomSampleMatches(
    const std::vector<cv::DMatch> &matches,
    int nSample);

std::vector<cv::DMatch> selectMatches(
    const std::vector<cv::DMatch> &matches,
    const VectorXb &mask);

std::tuple<Matrix2Xf, Matrix2Xf> getPixelPoints(
    const Frame::Ptr &frame1,
    const Frame::Ptr &frame2,
    const std::vector<cv::DMatch> &matches);

}

#endif //DTAM_COMMONUTIL_HPP
