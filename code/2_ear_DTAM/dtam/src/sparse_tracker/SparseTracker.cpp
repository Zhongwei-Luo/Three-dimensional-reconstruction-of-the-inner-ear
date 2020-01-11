//
// Created by 宋孝成 on 2019-05-01.
//

#include <sparse_tracker/utils/ScalePropagationUtil.hpp>
#include <sparse_tracker/utils/RelativePoseUtil.hpp>
#include "SparseTracker.hpp"
#include "feature_handler/FeatureHandler.hpp"
#include "utils/CommonUtil.hpp"

using namespace std;

namespace dtam {

SparseTracker::SparseTracker(
    Camera::Ptr camera,
    FeatureHandlerType featureHandlerType)
    : camera_(std::move(camera)),
      featureHandler_(FeatureHandler::create(featureHandlerType)) {}

bool SparseTracker::init(Frame::Ptr frame,
                         const Matrix44f &initTwc,
                         float initScale) {
  frameArray_.clear();
  featureHandler_->featureExtraction(frame);
  frame->setTwc(initTwc);
  frameArray_.emplace_back(frame);
  initScale_ = initScale;
  return true;
}

void SparseTracker::addFrame(Frame::Ptr frame, [[maybe_unused]] bool refine) {
  // extract relative pose
  featureHandler_->featureExtraction(frame);
  const auto &lastFrame = frameArray_.back();
  featureHandler_->featureMatching(lastFrame, frame);
  const auto matches = featureHandler_->getMatches(lastFrame, frame);
  //drawMatches(lastFrame, frame, matches);
  const auto cameraMat = camera_->getCameraMat();
  auto[R, t, mask] = extractRelativePoseOpenCV(lastFrame,
                                               frame,
                                               matches,
                                               cameraMat);
  Matrix44f T32;

  // scale propagation
  if (frameArray_.size() > 1) {
    const auto &frame1 = frameArray_.at(frameArray_.size() - 2);
    const auto &frame2 = lastFrame;
    const auto &frame3 = frame;
    const auto &m21 = featureHandler_->getMatches(frame2, frame1);
    const auto &m23 = featureHandler_->getMatches(frame2, frame3);
    const auto &K = camera_->getCameraMat();
    T32 = (Matrix44f() << R, t, 0, 0, 0, 1).finished();
    const auto scale = relativeScale(frame1, frame2, frame3, m21, m23, K, T32);
    t *= scale;
  } else {
    t *= initScale_ / t.norm();
  }

  cout << t.norm() << endl;

  // set the camera pose for this frame
  T32 = (Matrix44f() << R, t, 0, 0, 0, 1).finished();
  frame->setTcw(T32 * lastFrame->getTcw());

  // append to the frame list
  frameArray_.emplace_back(frame);
}

}