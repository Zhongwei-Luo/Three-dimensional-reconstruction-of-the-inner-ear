//
// Created by 宋孝成 on 2019-04-30.
//

#include <Frame.hpp>
#include "CostVolume.hpp"
#include "dense_mapper/utils/MathUtils.hpp"
#include "utils/ResultWritingUtil.hpp"
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include <iostream>

using namespace cv;

namespace dtam {

void CostVolume::reset() {
  referenceFrame_ = nullptr;
  frameCount_ = 0;
}

void CostVolume::setReferenceFrame(const Frame::Ptr &referenceFrame) {
  referenceFrame_ = referenceFrame;
  frameCount_++;
  const int rows = camera_->getWidth();
  const int cols = camera_->getHeight();
  costData_ = std::vector<MatrixXXf>(depthLayers_, 3*MatrixXXf::Ones(rows, cols));
  minCostInvDepth_ = MatrixXXf::Zero(rows, cols);
  minCost_ = MatrixXXf::Zero(rows, cols);
  maxCost_ = MatrixXXf::Zero(rows, cols);
}

void CostVolume::updateCost(const Frame::Ptr &currentFrame)

 {  //question
    //Twc=Twr,因为这里的c就是r，pw=Twc*pc (in homogenous)
    //Twr ，if I take the coordinate system of  referenceframe as the world  coordinate system
    //then Twr=I  ,Tmr=Tmw*Twr=Tmw;
  std::cout<<"--------------- "<<  std::endl;
     std::cout<<"begin update cost:  "<<  std::endl;



    const Matrix44f &Twr = referenceFrame_->getTwc();
  std::cout<<"Twr (should be I): "<<std::endl<<Twr<<std::endl;


  const Matrix44f &Tmw = currentFrame->getTcw();
  Matrix44f Tmr = Tmw * Twr;
  std::cout<<"Tmr (shold be equal to Tcw): "<<std::endl<<Tmr<<std::endl;
  const float invDepthStep =
      (maxInvDepth_ - minInvDepth_) / depthLayers_;
  const float fx = camera_->fx();
  const float fy = camera_->fy();
  const float cx = camera_->cx();
  const float cy = camera_->cy();
  const int width = camera_->getWidth();
  const int height = camera_->getHeight();

  // variables used in the loop
  float zr, xr, yr;
  Vector3f r, m;
  float um, vm;

  const int rows = camera_->getWidth();
  const int cols = camera_->getHeight();
  std::vector<MatrixXXf> cost(depthLayers_, MatrixXXf::Zero(rows, cols));

  for (int ur = 0; ur < width; ur++)
  {
    for (int vr = 0; vr < height; vr++)
    {//对每一个像素都这么做

      const Vector3f Ir = referenceFrame_->getRGBFloat(ur, vr);
      // TODO: why costMin=1?
      float costMin = 3, costMax = -1;

      int minCostLayer = 0;

        //总共depthLayers 层 depth costvolume
      for (int l = 0; l < depthLayers_; l++) {
        // reprojection
        const float d = minInvDepth_ + l * invDepthStep;
        //三维坐标
        zr = 1.0f / d;
        xr = (ur - cx) / fx * zr;
        yr = (vr - cy) / fy * zr;
        
        //m就是图片里面那个 m属于 L（r）集合的 m

        r = std::move((Vector3f(3, 1) << xr, yr, zr).finished());
        //从 r坐标系 转换到 m 坐标系
        
        m = Tmr.topLeftCorner(3, 3) * r + Tmr.col(3).head<3>();
        //对应m的图像坐标
        um = fx * (m.x() / m.z()) + cx;
        vm = fy * (m.y() / m.z()) + cy;
        // calculate reproj error (set to max difference if outside the frame)
        float rho; // = 3.0;
        if (m.z() > 0 && um >= 0 && um < width && vm >= 0 && vm < height) {

          const Vector3f Im = currentFrame->getRGBFloat(um, vm, width, height);
          rho = (Ir - Im).lpNorm<1>();
        } else {
            //costData_  在之前setReferenceFrame 已经初始化过了  
          rho = costData_[l](ur, vr);
        }
        cost[l](ur, vr) = rho;
        costData_[l](ur, vr) =
            (costData_[l](ur, vr) * (frameCount_ - 1) + rho) / frameCount_;

        // maintain min and max cost
        //will be use in update Eaux
        if (costData_[l](ur, vr) <= costMin) {
          costMin = costData_[l](ur, vr);
          minCostLayer = l;
        }
        if (costData_[l](ur, vr) > costMax) {
          costMax = costData_[l](ur, vr);
        }
      }
       //fininal,there are a mincost and maxcost matrix
      minCost_(ur, vr) = costMin;
      minCostInvDepth_(ur, vr) = minInvDepth_ + minCostLayer * invDepthStep;
      maxCost_(ur, vr) = costMax;

    }
  }
  std::cout << minCostInvDepth_.maxCoeff() << " " << minCostInvDepth_.minCoeff() << std::endl;

  saveCostVolume(cost, std::to_string(frameCount_));
  frameCount_++;

     std::cout<<"end update cost:  "<<  std::endl;   std::cout<<"--------------- "<<  std::endl;
}

void CostVolume::updateCost(const std::string& filepath) {
  cv::Mat invdepth;
  cv::FileStorage storage(filepath, cv::FileStorage::READ);
  if (storage.isOpened()) {
    storage["img"] >> invdepth;
    storage.release();
  }

  MatrixXXf m;
  cv::cv2eigen(invdepth, m);
  minCostInvDepth_ = m.transpose();

  frameCount_++;
}

}
