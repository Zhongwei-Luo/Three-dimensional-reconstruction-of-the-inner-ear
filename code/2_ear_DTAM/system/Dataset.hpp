//
// Created by 宋孝成 on 2019-05-04.
//

#ifndef DTAM_DATASET_HPP
#define DTAM_DATASET_HPP

#include <limits>
#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/mat.hpp>
#include <experimental/filesystem>
#include <utils/MatrixType.hpp>
#include <Eigen/Dense>

#include <opencv2/core/eigen.hpp>


class Dataset {

 public:

  using Ptr = std::shared_ptr<Dataset>;

  static Dataset::Ptr create(
      const std::string &path,
      int maxNumber = std::numeric_limits<int>::max()) {
    return Dataset::Ptr(new Dataset(path, maxNumber));
  }

  Dataset(const std::string &path, int maxNumber)
      : path_(path), maxNumber_(maxNumber) {};

  void reset();
  bool back();
  bool next();
  double readTimestamp() const;
  cv::Mat readRGB() const;
    Eigen::Matrix4f readTwc(std:: vector<dtam::cv_T> &vec) const;
    size_t getNumber() const { return index_; }

 protected:

  std::string getMainFileName() const;

  const std::experimental::filesystem::path path_;

  size_t index_ = -1;
  const size_t maxNumber_;

};

#endif //DTAM_DATASET_HPP
