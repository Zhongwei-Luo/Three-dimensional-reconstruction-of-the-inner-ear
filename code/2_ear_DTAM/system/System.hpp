//
// Created by 宋孝成 on 2019-04-30.
//

#ifndef DTAM_SYSTEM_HPP
#define DTAM_SYSTEM_HPP

#include <DenseMapper.hpp>
#include <Camera.hpp>
#include <thread>
#include <SparseTracker.hpp>
#include "Dataset.hpp"
#include "utils/MatrixType.hpp"
class System
        {

 public:
 vector<dtam::cv_T> v_poses_;
  System(const std::string &dataFolderPath,
         const std::string &settingsFilePath,
         std::string &trac);

  void start();
  void stop();

  void showPointCloud();
  void showDepthImage();
  void dtam();

 protected:

  dtam::DenseMapper::Ptr denseMapper_ = nullptr;
  dtam::SparseTracker::Ptr sparseTracker_ = nullptr;

  bool updatePCPending_ = false;
  bool updateDepthPending_ = false;
  bool stopped_ = false;
  Dataset::Ptr dataset_ = nullptr;
    string trajectory_file_path_;
};

#endif //DTAM_SYSTEM_HPP
