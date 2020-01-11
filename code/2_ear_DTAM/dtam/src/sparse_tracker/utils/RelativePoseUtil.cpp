//
// Created by songxiaocheng on 19-5-10.
//

#include "RelativePoseUtil.hpp"
#include "CommonUtil.hpp"
#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>

using namespace std;

namespace dtam {

tuple<Matrix33f, Vector3f, VectorXb> extractRelativePoseOpenCV(
    const Frame::Ptr &frame1,
    const Frame::Ptr &frame2,
    const std::vector<cv::DMatch> &matches,
    const Matrix33f &cameraMat) {
  cv::Mat cK;
  eigen2cv(cameraMat, cK);
  vector<cv::Point2f> points1;
  vector<cv::Point2f> points2;
  for (const auto &match:matches) {
    points1.emplace_back(frame1->getKeyPoints().at(match.queryIdx).pt);
    points2.emplace_back(frame2->getKeyPoints().at(match.trainIdx).pt);
  }
  cv::Mat cmask;
  const auto E = cv::findEssentialMat(
      points1, points2, cK, cv::RANSAC, 0.999, 1, cmask);
  cout << "inliers after ransac: " << cv::countNonZero(cmask) << endl;
  cout << E << endl;
  vector<cv::Point2f> inliers1, inliers2;
  vector<cv::DMatch> inlierMatches;
  for (int i = 0; i < cmask.rows; i++) {
    if (cmask.at<unsigned char>(i)) {
      inlierMatches.emplace_back(matches[i]);
      inliers1.emplace_back(points1[i]);
      inliers2.emplace_back(points2[i]);
    }
  }
  VectorXb mask;
  cv2eigen(cmask, mask);
  cmask.release();
  cv::Mat cR, ct;
  //cv::correctMatches(E, inliers1, inliers2, inliers1, inliers2);
  cv::recoverPose(E, inliers1, inliers2, cK, cR, ct, cmask);
  cout << "inliers after recover pose: " << cv::countNonZero(cmask) << endl;
  Matrix33f R;
  Vector3f t;
  cv2eigen(cR, R);
  cv2eigen(ct, t);
  return {R, t, mask};
}

const inline Matrix33f getW() {
  static const Matrix33f
      W = (Matrix33f() << 0, -1, 0, 1, 0, 0, 0, 0, 1).finished();
  return W;
}

const inline Matrix33f getWt() {
  static const Matrix33f
      Wt = (Matrix33f() << 0, 1, 0, -1, 0, 0, 0, 0, 1).finished();
  return Wt;
}

std::tuple<Matrix33f, Matrix33f, Vector3f> decomposeEssentialMatrix(
    const Matrix33f &essentialMat) {
  Eigen::JacobiSVD<Matrix33f>
      svd(essentialMat, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Matrix33f &Vt = svd.matrixV().transpose(), &U = svd.matrixU();
  const auto t = U.col(2);
  const auto R1 = U * getW() * Vt;
  const auto R2 = U * getWt() * Vt;
  return {R1, R2, t};
}

tuple<Matrix33f, Vector3f, Matrix3Xf> recoverPose(
    const Matrix2Xf &cameraPoints1,
    const Matrix2Xf &cameraPoints2,
    const Matrix33f &essentialMat) {
  const auto[R1, R2, t] = decomposeEssentialMatrix(essentialMat);
  const auto P1 = Matrix34f::Identity();
  Matrix34f P2_1, P2_2, P2_3, P2_4;
  P2_1 << R1, t;
  P2_2 << R1, -t;
  P2_3 << R2, t;
  P2_4 << R2, -t;
  const auto ps3d_1 = triangulatePoints(cameraPoints1, cameraPoints2, P1, P2_1);
  const auto ps3d_2 = triangulatePoints(cameraPoints1, cameraPoints2, P1, P2_2);
  const auto ps3d_3 = triangulatePoints(cameraPoints1, cameraPoints2, P1, P2_3);
  const auto ps3d_4 = triangulatePoints(cameraPoints1, cameraPoints2, P1, P2_4);
  int c1 = 0, c2 = 0, c3 = 0, c4 = 0;
  const auto reproj2d_1 = P2_1 * ps3d_1.colwise().homogeneous();
  const auto reproj2d_2 = P2_2 * ps3d_2.colwise().homogeneous();
  const auto reproj2d_3 = P2_3 * ps3d_3.colwise().homogeneous();
  const auto reproj2d_4 = P2_4 * ps3d_4.colwise().homogeneous();
  for (int i = 0; i < cameraPoints1.rows(); i++) {
    if (ps3d_1(i, 2) > 0 && reproj2d_1(i, 2) > 0) c1++;
    if (ps3d_2(i, 2) > 0 && reproj2d_2(i, 2) > 0) c2++;
    if (ps3d_3(i, 2) > 0 && reproj2d_3(i, 2) > 0) c3++;
    if (ps3d_4(i, 2) > 0 && reproj2d_4(i, 2) > 0) c4++;
  }
  const auto max_count = std::max({c1, c2, c3, c4});
  Matrix3Xf points3d;
  Matrix33f rotation;
  Vector3f translation;
  if (max_count == c1) {
    rotation = R1;
    translation = t;
    points3d = ps3d_1;
  } else if (max_count == c2) {
    rotation = R1;
    translation = -t;
    points3d = ps3d_2;
  } else if (max_count == c3) {
    rotation = R2;
    translation = t;
    points3d = ps3d_3;
  } else if (max_count == c4) {
    rotation = R2;
    translation = -t;
    points3d = ps3d_4;
  }
  return {rotation, translation, points3d};
}

vector<Matrix33f> computeFundamental7Pt(
    const Matrix2Xf &pixelSample1,
    const Matrix2Xf &pixelSample2) {
  const auto X = pixelSample1.row(0).transpose().array();
  const auto Y = pixelSample1.row(1).transpose().array();
  const auto Xp = pixelSample2.row(0).transpose().array();
  const auto Yp = pixelSample2.row(1).transpose().array();
  MatrixXXf A(7, 9);
  A << Xp * X, Xp * Y, Xp, Yp * X, Yp * Y, Yp, X, Y, MatrixXXf::Ones(7, 1);
  Eigen::JacobiSVD<MatrixXXf> svd(A, Eigen::ComputeFullV);
  const MatrixXXf &V = svd.matrixV();
  VectorXf V8 = V.col(7);
  VectorXf V9 = V.col(8);
  using Map33Row = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>;
  const Matrix33f F1 = Map33Row(V8.data());
  const Matrix33f F2 = Map33Row(V9.data());
  const VectorXc lambdas = (F2.inverse() * F1).eigenvalues();
  std::vector<Matrix33f> FArr;
  for (int i = 0; i < lambdas.size(); i++) {
    if (lambdas(i).imag() == 0) {
      FArr.emplace_back(F1 + lambdas(i).real() * F2);
    }
  }
  return FArr;
}

Matrix33f computeFundamental8Pt(
    const Matrix2Xf &pixelPoints1,
    const Matrix2Xf &pixelPoints2) {
  const auto X = pixelPoints1.row(0).transpose().array();
  const auto Y = pixelPoints1.row(1).transpose().array();
  const auto Xp = pixelPoints2.row(0).transpose().array();
  const auto Yp = pixelPoints2.row(1).transpose().array();
  const int n = pixelPoints1.cols();
  MatrixXXf A(n, 9);
  A << Xp * X, Xp * Y, Xp, Yp * X, Yp * Y, Yp, X, Y, MatrixXXf::Ones(n, 1);
  Eigen::JacobiSVD<MatrixXXf> svd(A, Eigen::ComputeFullV);
  const auto f = svd.matrixV().rightCols(1);
  Matrix33f F;
  F << f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8);
  return F;
}

tuple<Matrix33f, Vector3f, VectorXb> extractRelativePose(
    const Frame::Ptr &frame1,
    const Frame::Ptr &frame2,
    const std::vector<cv::DMatch> &matches,
    const Matrix33f &cameraMat) {

  float pixelThreshold = 1;
  int maxInlier = 0;
  VectorXb bestMask;
  const auto[pixPts1, pixPts2] = getPixelPoints(frame1, frame2, matches);
  const auto camPts1 = pixel2camera(pixPts1, cameraMat);
  const auto camPts2 = pixel2camera(pixPts2, cameraMat);
  const int n = matches.size();
  for (int i = 0; i < 100; i++) {
    const auto subMatches = randomSampleMatches(matches, 7);
    const auto[sample1, sample2] = getPixelPoints(frame1, frame2, subMatches);
    const auto FArr = computeFundamental7Pt(sample1, sample2);
    for (const auto &fundamentalMat: FArr) {
      const auto E = cameraMat.transpose() * fundamentalMat * cameraMat;
      const auto[R2, t2, p3d] = recoverPose(camPts1, camPts2, E);
      const auto R1 = Matrix33f::Identity();
      const auto t1 = Vector3f::Zero();
      const Matrix2Xf delta1 = pixPts1 - space2pixel(p3d, cameraMat, R1, t1);
      const Matrix2Xf delta2 = pixPts2 - space2pixel(p3d, cameraMat, R2, t2);
      VectorXb mask(n);
      for (int j = 0; j < n; j++) {
        mask(j) = ((delta1.col(j).norm() < pixelThreshold)
            && (delta2.col(j).norm() < pixelThreshold));
      }
      const int nInliers = mask.count();
      if (nInliers > maxInlier) {
        maxInlier = nInliers;
        bestMask = mask;
      }
    }
  }
  const auto bestMatches = selectMatches(matches, bestMask);
  const auto[inliers1, inliers2] = getPixelPoints(frame1, frame2, bestMatches);

  std::cout << maxInlier << std::endl;
  const auto width = frame1->getWidth();
  const auto height = frame1->getHeight();
  const auto normedInliers1 = normalizedPixels(inliers1, width, height);
  const auto normedInliers2 = normalizedPixels(inliers2, width, height);
  const auto F = computeFundamental8Pt(normedInliers1, normedInliers2);
  const auto inlierCameraPts1 = pixel2camera(inliers1, cameraMat);
  const auto inlierCameraPts2 = pixel2camera(inliers2, cameraMat);
  const auto E = cameraMat.transpose() * F * cameraMat;
  const auto[R, t, p3d] = recoverPose(inlierCameraPts1, inlierCameraPts2, E);
  return {R, t, bestMask};
}

}