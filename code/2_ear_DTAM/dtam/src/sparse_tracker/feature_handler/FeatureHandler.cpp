//
// Created by songxiaocheng on 19-5-9.
//

#include "FeatureHandler.hpp"

namespace dtam {

FeatureHandler::Ptr FeatureHandler::create(FeatureHandlerType type) {
  FeatureHandler::Ptr ptr;
  switch (type) {
    case ORBBF:ptr = std::make_shared<ORBBFFeatureHandler>();
      break;
    default:throw std::invalid_argument("feature handler type is unrecognized");
  }
  return ptr;
}

void FeatureHandler::featureExtraction(const Frame::Ptr &frame) {
  std::vector<cv::KeyPoint> keyPoints;
  cv::Mat descriptors;
  const auto &image = frame->getImageRGBUint8();
  detector_->detectAndCompute(image, cv::noArray(), keyPoints, descriptors);
  cout << "key points: " << keyPoints.size() << endl;
  frame->setFeature(keyPoints, descriptors);
}

const std::vector<cv::DMatch> &FeatureHandler::getMatches(
    const Frame::Ptr &frame1,
    const Frame::Ptr &frame2) {
  const pair key = {frame1, frame2};
  if (matchesMap_.find(key) == matchesMap_.end()) {
    featureMatching(frame1, frame2);
  }
  return matchesMap_.at(key);
}

void FeatureHandler::addMatches(const Frame::Ptr &frame1,
                                const Frame::Ptr &frame2,
                                const std::vector<cv::DMatch> &matches) {
  matchesMap_[{frame1, frame2}] = matches;
}

void FeatureHandler::clear() { matchesMap_.clear(); }

}