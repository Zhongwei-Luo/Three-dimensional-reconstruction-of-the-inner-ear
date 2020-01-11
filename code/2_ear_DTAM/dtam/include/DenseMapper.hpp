//
// Created by 宋孝成 on 2019-04-30.
//

#ifndef DTAM_DENSEMAPPER_HPP
#define DTAM_DENSEMAPPER_HPP

#include <memory>
#include <mutex>
#include <opencv2/core/core.hpp>
#include "Camera.hpp"
#include "Frame.hpp"

namespace dtam {

class CostVolume;
class Optimizer;

class DenseMapper {

 public:

  using Ptr = std::shared_ptr<DenseMapper>;

  static DenseMapper::Ptr create

      (
        const Camera::Ptr &camera,
         const cv::FileStorage &settings)
      {
    return std::make_shared<DenseMapper>(camera, settings);
  }

  DenseMapper(Camera::Ptr camera, const cv::FileStorage &settings);

  cv::Mat getRefDepthImage();
  cv::Mat getRefInvDepthImage();

  void addFrame(const Frame::Ptr &frame);

  const std::vector<MatrixXXf> &getCosts() const;

 protected:

  const Camera::Ptr camera_;

  std::shared_ptr<CostVolume> costVolume_ = nullptr;
  std::shared_ptr<Optimizer> optimizer_ = nullptr;
  Frame::Ptr refFrame_;
  MatrixXXf refInvDepth_;

  int maxFramesPerCostVolume_;
  int nIters_;
  std::mutex updateDepthMutex_;

};

}

#endif //DTAM_DENSEMAPPER_HPP
