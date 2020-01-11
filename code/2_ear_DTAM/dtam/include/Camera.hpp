//
// Created by 宋孝成 on 2019-04-30.
//

#ifndef DTAM_CAMERA_HPP
#define DTAM_CAMERA_HPP

#include <memory>
#include "utils/MatrixType.hpp"

namespace dtam {

class Camera {
 public:
  using Ptr = std::shared_ptr<Camera>;

  static Camera::Ptr create(
      int width, int height, const Matrix33f &cameraMatrix) {
    return std::make_shared<Camera>(width, height, cameraMatrix);
  }

  static Camera::Ptr create(
      int width, int height, float fx, float fy, float cx, float cy) {
    const auto K = (Matrix33f() << fx, 0, cx, 0, fy, cy, 0, 0, 1).finished();
    return std::make_shared<Camera>(width, height, K);
  }

  Camera(int width, int height, Matrix33f cameraMatrix)
      : width_(width),
        height_(height),
        cameraMatrix_(std::move(cameraMatrix)) {};

  const Matrix33f &getCameraMat() const { return cameraMatrix_; }

  const float &fx() const { return cameraMatrix_(0, 0); }

  const float &fy() const { return cameraMatrix_(1, 1); }

  const float &cx() const { return cameraMatrix_(0, 2); }

  const float &cy() const { return cameraMatrix_(1, 2); }

  const int &getWidth() const { return width_; }

  const int &getHeight() const { return height_; }

 protected:

  const int width_;
  const int height_;
  const Matrix33f cameraMatrix_;

};

}

#endif //DTAM_CAMERA_HPP
