//
// Created by 宋孝成 on 2019-04-14.
//

#ifndef DTAM_RESULTWRITINGUTIL_HPP
#define DTAM_RESULTWRITINGUTIL_HPP

#include <cnpy.h>
#include <iostream>
#include <vector>
#include "utils/MatrixType.hpp"
#include "Frame.hpp"
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>

namespace {

namespace fs = std::experimental::filesystem;

const inline std::string getDepthFolder() {
  static const std::string
      folder = fs::current_path().string() + "/result/depth/";
  return folder;
}

const inline std::string getCostVolumeFolder() {
  static const std::string
      folder = fs::current_path().string() + "/result/cost_volume/";
  return folder;
}

const inline std::string getTrajectoryFolder() {
  static const std::string
      folder = fs::current_path().string() + "/result/trajectory/";
  return folder;
}

void createFolderIfAbsent(std::string folderName) {
  if (!fs::exists(fs::path(folderName))) {
    fs::create_directories(folderName);
  }
}

void saveCostVolume(
    const std::vector<dtam::MatrixXXf> &costVolume,
    const std::string &name) {
  unsigned long nz = costVolume.size();
  assert(nz > 0);
  unsigned long nx = costVolume[0].rows();
  unsigned long ny = costVolume[0].cols();
  std::vector<float> data;
  data.reserve(nx * ny * nz);

  for (int i = 0; i < nx; i++) {
    for (int j = 0; j < ny; j++) {
      for (int k = 0; k < nz; k++) {
        const auto &costs = costVolume[k];
        data.emplace_back(costs(i, j));
      }
    }
  }
  const auto filePath = getCostVolumeFolder() + name + ".npy";
  createFolderIfAbsent(getCostVolumeFolder());
  cnpy::npy_save(filePath, data.data(), {nx, ny, nz}, "w");
}

void saveDepthData(const cv::Mat &img, const std::string &name) {
  createFolderIfAbsent(getDepthFolder());
  std::string yamlf = getDepthFolder() + name + ".yaml";
  cv::FileStorage storage(yamlf, cv::FileStorage::WRITE);
  storage << "img" << img;
  storage.release();
}

void saveDepthImage(const cv::Mat &normalizedImg, const std::string &name) {
  createFolderIfAbsent(getDepthFolder());
  std::string exrf = getDepthFolder() + name + ".exr";
  std::string pngf = getDepthFolder() + name + ".png";
  cv::imwrite(exrf, normalizedImg);
  cv::imwrite(pngf, 255 * normalizedImg);
}

void clearTrajectory(const std::string &name) {
  const auto filePath = getTrajectoryFolder() + name + ".npy";
  if (fs::exists(fs::path(filePath))) {
    fs::remove(filePath);
  }
}

void appendTrajectory(const dtam::Matrix44f &twc, const std::string &name) {
  std::vector<float> data;
  const dtam::Vector3f position = twc.col(3).head(3);
  const dtam::Matrix33f rotation = twc.topLeftCorner(3, 3);
  const auto orientation = dtam::Quaternionf(rotation);
  data.emplace_back(position.x());
  data.emplace_back(position.y());
  data.emplace_back(position.z());
  data.emplace_back(orientation.x());
  data.emplace_back(orientation.y());
  data.emplace_back(orientation.z());
  data.emplace_back(orientation.w());
  const auto filePath = getTrajectoryFolder() + name + ".npy";
  createFolderIfAbsent(getTrajectoryFolder());
  if (!fs::exists(fs::path(filePath))) {
    cnpy::npy_save(filePath, data.data(), {1, 7}, "w");
  } else {
    cnpy::npy_save(filePath, data.data(), {1, 7}, "a");
  }
}

}

#endif //DTAM_RESULTWRITINGUTIL_HPP
