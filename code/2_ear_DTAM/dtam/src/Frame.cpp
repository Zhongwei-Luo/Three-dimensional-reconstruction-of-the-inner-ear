//
// Created by 宋孝成 on 2019-04-30.
//

#include "Frame.hpp"

namespace dtam {

void Frame::setTcw(const Matrix44f &tcw) {
  tcw_ = tcw;
  const Matrix33f rot_wc = tcw.topLeftCorner(3, 3).transpose();
  const Vector3f tr_wc = -rot_wc * tcw.col(3).head<3>();
  twc_ = (Matrix44f() << rot_wc, tr_wc, 0, 0, 0, 1).finished();
}

void Frame::setTwc(const Matrix44f &twc)
{
  twc_ = twc;
  const Matrix33f rot_cw = twc.topLeftCorner(3, 3).transpose();
  const Vector3f tr_cw = -rot_cw * twc.col(3).head<3>();
  tcw_ = (Matrix44f() << rot_cw, tr_cw, 0, 0, 0, 1).finished();
}

}