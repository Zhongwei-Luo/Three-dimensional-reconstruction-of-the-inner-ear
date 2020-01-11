//
// Created by songxiaocheng on 19-5-9.
//

#include "ScalePropagationUtil.hpp"
#include "CommonUtil.hpp"
#include <opencv2/core/eigen.hpp>

using namespace std;

namespace dtam {

tuple<vector<cv::Point2d>,
      vector<cv::Point2d>,
      vector<cv::Point2d>>
findCommonPoints(
    const Frame::Ptr &frame1,
    const Frame::Ptr &frame2,
    const Frame::Ptr &frame3,
    const vector<cv::DMatch> &matches21,
    const vector<cv::DMatch> &matches23) {

  vector<tuple<int, int, int>> triples;
  map<int, int> map21;
  map<int, int> map23;
  for (const auto &match: matches21) {
    map21[match.queryIdx] = match.trainIdx;
  }
  for (const auto &match: matches23) {
    map23[match.queryIdx] = match.trainIdx;
  }
  for (const auto &entry: map21) {
    const auto search = map23.find(entry.first);
    if (search != map23.end()) {
      // in {frame 1, frame 2, frame 3} order
      const auto &t = make_tuple(entry.second, entry.first, search->second);
      triples.emplace_back(t);
    }
  }

  vector<cv::Point2d> points1, points2, points3;
  for (const auto[i1, i2, i3]: triples) {
    points1.emplace_back(frame1->getKeyPoints()[i1].pt);
    points2.emplace_back(frame2->getKeyPoints()[i2].pt);
    points3.emplace_back(frame3->getKeyPoints()[i3].pt);
  }
  return {points1, points2, points3};
}

float relativeScale(
    const Frame::Ptr &frame1,
    const Frame::Ptr &frame2,
    const Frame::Ptr &frame3,
    const std::vector<cv::DMatch> &matches21,
    const std::vector<cv::DMatch> &matches23,
    const Matrix33f &cameraMat,
    const Matrix44f &T32) {
  const auto[points1, points2, points3] = findCommonPoints(
      frame1, frame2, frame3, matches21, matches23);
  const Matrix44f T12 = frame1->getTcw() * frame2->getTwc();
  std::vector<Vector3f> p3d21, p3d23;
  p3d21 = relativeTriangulate(points2, points1, cameraMat, T12);
  p3d23 = relativeTriangulate(points2, points3, cameraMat, T32);
  std::vector<float> ds;
  for (size_t i = 0; i < points2.size(); i++) {
    ds.emplace_back(p3d21.at(i).norm() / p3d23.at(i).norm());
    cout << ds.back() << ",";
  }
  cout << endl;
  std::nth_element(ds.begin(), ds.begin() + ds.size() / 2, ds.end());
  return ds[ds.size() / 2];
}

}
