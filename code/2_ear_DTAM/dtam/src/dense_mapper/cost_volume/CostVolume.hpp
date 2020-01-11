//
// Created by 宋孝成 on 2019-04-30.
//

#ifndef DTAM_COSTVOLUME_HPP
#define DTAM_COSTVOLUME_HPP

#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include "Camera.hpp"
#include "Frame.hpp"

namespace dtam {

class CostVolume {

 public:

  using Ptr = std::shared_ptr<CostVolume>;

  static CostVolume::Ptr create(const Camera::Ptr &camera,
                                int depthLayers,
                                float minInvDepth,
                                float maxInvDepth) {
    return std::make_shared<CostVolume>(camera,
                                        depthLayers,
                                        minInvDepth,
                                        maxInvDepth);
  }

  CostVolume(Camera::Ptr camera,
             int depthLayers,
             float minInvDepth,
             float maxInvDepth
  ) : camera_(std::move(camera)),
      depthLayers_(depthLayers),
      minInvDepth_(minInvDepth),
      maxInvDepth_(maxInvDepth) {
    reset();
  };

  void setReferenceFrame(const Frame::Ptr &referenceFrame);

  void reset();

  void updateCost(const Frame::Ptr &currentFrame);
  void updateCost(const std::string &filepath);

  const std::vector<MatrixXXf> &getCosts() const { return costData_; }
  const MatrixXXf &getMinCost() const { return minCost_; }
  const MatrixXXf &getMaxCost() const { return maxCost_; }
  const MatrixXXf &getMinCostInvDepth() const { return minCostInvDepth_; }

  float getMinInvDepth() const { return minInvDepth_; }
  float getMaxInvDepth() const { return maxInvDepth_; }
  int getDepthLayers() const { return depthLayers_; }
  int getFrameLayer() const { return frameCount_; }

  float getMinDepth() const { return 1.0f / minCostInvDepth_.maxCoeff(); }

 protected:

  const Camera::Ptr camera_;

  std::vector<MatrixXXf> costData_ = {};
  MatrixXXf minCostInvDepth_;
  MatrixXXf minCost_;
  MatrixXXf maxCost_;
  int frameCount_ = 0;
  Frame::Ptr referenceFrame_ = nullptr;

  const int depthLayers_;
  const float minInvDepth_;
  const float maxInvDepth_;

};

}

#endif //DTAM_COSTVOLUME_HPP
