//
// Created by 宋孝成 on 2019-05-01.
//

#ifndef DTAM_SPARSETRACKER_HPP
#define DTAM_SPARSETRACKER_HPP

#include <memory>
#include "utils/MatrixType.hpp"
#include "Frame.hpp"
#include <opencv2/opencv.hpp>

using namespace std;

namespace dtam {

class FeatureHandler;

enum FeatureHandlerType { ORBBF };

class SparseTracker {

 public:

  using Ptr = std::shared_ptr<SparseTracker>;

  static SparseTracker::Ptr create(
      const Camera::Ptr &camera,
      FeatureHandlerType featureHandlerType) {
    return std::make_shared<SparseTracker>(camera, featureHandlerType);
  }

  SparseTracker(Camera::Ptr camera, FeatureHandlerType featureHandlerType);

  bool init(Frame::Ptr frame, const Matrix44f &initTwc, float initScale = 1);

  void addFrame(Frame::Ptr frame, [[maybe_unused]] bool refine = false);

  const std::vector<Frame::Ptr> &getFrames() const { return frameArray_; }

 protected:

  std::vector<Frame::Ptr> frameArray_;

  float initScale_ = 1;

  const Camera::Ptr camera_;
  const std::shared_ptr<FeatureHandler> featureHandler_;

};

}

#endif //DTAM_SPARSETRACKER_HPP
