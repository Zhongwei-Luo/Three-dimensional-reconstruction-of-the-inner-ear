//
// Created by SongXiaocheng on 4/12/2019.
//

#ifndef DTAM_FEATUREHANDLER_HPP
#define DTAM_FEATUREHANDLER_HPP

#include <utils/MatrixType.hpp>
#include "Frame.hpp"
#include <map>
#include "SparseTracker.hpp"

namespace dtam {

class FeatureHandler {

  using pair = std::pair<Frame::Ptr, Frame::Ptr>;

 public:

  using Ptr = std::shared_ptr<FeatureHandler>;

  static FeatureHandler::Ptr create(FeatureHandlerType type);

  FeatureHandler() = default;

  void featureExtraction(const Frame::Ptr &frame);

  virtual void featureMatching(const Frame::Ptr &frame1,
                               const Frame::Ptr &frame2) = 0;

  const std::vector<cv::DMatch> &getMatches(
      const Frame::Ptr &frame1,
      const Frame::Ptr &frame2);

  void clear();

 protected:

  void addMatches(const Frame::Ptr &frame1,
                  const Frame::Ptr &frame2,
                  const std::vector<cv::DMatch> &matches);

  cv::Ptr<cv::Feature2D> detector_;
  cv::Ptr<cv::DescriptorMatcher> matcher_;
  std::map<pair, std::vector<cv::DMatch>> matchesMap_;

};

class ORBBFFeatureHandler : public FeatureHandler {

 public:

  ORBBFFeatureHandler();
  void featureMatching(const Frame::Ptr &frame1,
                       const Frame::Ptr &frame2) override;

};

}
#endif //DTAM_FEATUREHANDLER_HPP
