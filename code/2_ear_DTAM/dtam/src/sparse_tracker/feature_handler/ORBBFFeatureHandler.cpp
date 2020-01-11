//
// Created by 宋孝成 on 2019-04-22.
//

#ifndef BUNDLEADJUSTMENT_ORBBFFEATUREHANDLER_HPP
#define BUNDLEADJUSTMENT_ORBBFFEATUREHANDLER_HPP

#include "FeatureHandler.hpp"

namespace dtam {

ORBBFFeatureHandler::ORBBFFeatureHandler() {
  detector_ = cv::ORB::create(10000);
  matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING2);
}

void ORBBFFeatureHandler::featureMatching(
    const Frame::Ptr &frame1,
    const Frame::Ptr &frame2) {

  const auto &des1 = frame1->getDescriptor();
  const auto &des2 = frame2->getDescriptor();

  std::vector<std::vector<cv::DMatch> > matches2NN;
  matcher_->knnMatch(des1, des2, matches2NN, 2);
  std::cout << matches2NN.size() << std::endl;
  const double ratio = 0.5;
  std::vector<cv::DMatch> goodMatches;
  for (size_t i = 0; i < matches2NN.size(); i++) { // i is queryIdx
    if (matches2NN[i][0].distance / matches2NN[i][1].distance < ratio) {
      goodMatches.emplace_back(matches2NN[i][0]);
    }
  }
  std::cout << goodMatches.size() << std::endl;

  cout << "matchings: " << goodMatches.size() << endl;
  addMatches(frame1, frame2, goodMatches);

}

}

#endif //BUNDLEADJUSTMENT_ORBBFFEATUREHANDLER_HPP
