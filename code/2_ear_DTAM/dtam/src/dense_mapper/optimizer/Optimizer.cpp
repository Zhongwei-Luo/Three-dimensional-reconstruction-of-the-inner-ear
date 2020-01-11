//
// Created by 宋孝成 on 2019-04-30.
//

#include "dense_mapper/cost_volume/CostVolume.hpp"
#include "dense_mapper/utils/MathUtils.hpp"
#include "Optimizer.hpp"

#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

namespace dtam
{

void Optimizer::initByReferenceFrame(const Frame::Ptr &referenceFrame) 
{
  qx_ = MatrixXXf::Zero(width_, height_);
  qy_ = MatrixXXf::Zero(width_, height_);
  g_ = MatrixXXf::Zero(width_, height_);
  d_ = MatrixXXf::Zero(width_, height_);
  a_ = d_;//question : d_ a_ should be start with argmin C(u,a_{u})?
  computeG(referenceFrame);
  computeL();//L_ 是什么
}

void Optimizer::computeG(const Frame::Ptr &referenceFrame) {
  for (int u = 0; u < width_; u++)
  {
    for (int v = 0; v < height_; v++)
    {
      // gradients gx := $\partial_{x}^{+}img$ computed using forward differences
      const float right = referenceFrame->getGrayFloat(u + 1, v);
      const float bottom = referenceFrame->getGrayFloat(u, v + 1);
      const float center = referenceFrame->getGrayFloat(u, v);
      float gx = (u == width_ - 1) ? 0.0f : right - center;
      float gy = (v == height_ - 1) ? 0.0f : bottom - center;
      g_(u, v) = expf(-alphaG_ * powf(sqrtf(gx * gx + gy * gy), betaG_));
    }
  }
}

void Optimizer::computeL() {
  L_ = 2.0;//L_ 是什么
}

void Optimizer::optimize(int nIters) {
  int n = 1;
  theta_ = thetaBegin_;
  d_ = costVolume_->getMinCostInvDepth();
  a_ = d_;
  bool isFullyOpt = nIters == 0;
  isFullyOpt = true;
  // If fully optimized, use theta criterion, otherwise use nIters criterion.
  while (isFullyOpt ? (theta_ > thetaEnd_) : (n < nIters))
  {
    computeSigmas();
    //std::cout << "max_a: " << a_.maxCoeff() << std::endl;
    //std::cout << "min_a: " << a_.minCoeff() << std::endl;
    //cout << "\t\t on updating Q/D" << endl;

    //question: in the paper,they optimise q,and then d,and then a
    //why in here,we optimise q,d,for i  times
    for (int i = 0; i < 50; ++i) {
      updateQ();
      updateD();
      cv::Mat cvd_;
      cv::Mat normalized_d;

      //qusetion :why use transpose()
      eigen2cv(d_.transpose().eval(), cvd_);
      cv::normalize(cvd_, normalized_d, 1, 0, cv::NORM_MINMAX);
      cv::imshow("d_", normalized_d);
      cv::waitKey(10);
    }

    //cout << "\t\t on updating A" << endl;

    updateA();
    // decrement theta
    float beta = (theta_ > 1e-3f) ? 1e-3f : 1e-4f;
    theta_ *= 1 - beta * n;
    n++;

    cv::Mat cva_;
    cv::Mat normalized_a;
    eigen2cv(a_.transpose().eval(), cva_);
    cv::normalize(cva_, normalized_a, 1, 0, cv::NORM_MINMAX);
    cv::imshow("a_", normalized_a);
    cv::waitKey(10);

    //cout << "theta: " << theta_ << endl;
  }
  //cout<<"before finished"<<endl;
 // getchar();
}

void Optimizer::computeSigmas() {
  float mu = 2.0f * std::sqrt(epsilon_ / theta_) / L_;
  sigmaD_ = mu / (2.0f / theta_);
  sigmaQ_ = mu / (2.0f * epsilon_);
}

void Optimizer::updateQ() {
  const float denominator = 1.0f + sigmaQ_ * epsilon_;
  for (int u = 0; u < width_; u++) {
    for (int v = 0; v < height_; v++) {
        //dd_x.dd_y就是计算梯度，就是学长report上的(2.5)

        //question:学长是如何知道 q 有两个分量的，我感觉在paper中看出来这一点是非常难的一件事情
        //在paper中q就是一个非常抽象的量
      const float dd_x = (u == width_ - 1) ? 0.0f : d_(u + 1, v) - d_(u, v);
      const float dd_y = (v == height_ - 1) ? 0.0f : d_(u, v + 1) - d_(u, v);
      const float qx = (qx_(u, v) + sigmaQ_ * g_(u, v) * dd_x) / denominator;
      const float qy = (qy_(u, v) + sigmaQ_ * g_(u, v) * dd_y) / denominator;
      const float projectQ = max(1.0f, sqrtf(qx * qx + qy * qy));
      qx_(u, v) = qx / projectQ;
      qy_(u, v) = qy / projectQ;
    }
  }
}

void Optimizer::updateD() {
  const float denominator = 1.0f + sigmaD_ / theta_;
  for (int u = 0; u < width_; u++)
  {
    for (int v = 0; v < height_; v++)
    {//这一块，不能理解，虽然是论文中的2.23部分第一步优化过程
      const float dqx_x =
          (u == 0) ? qx_(u, v) - qx_(u + 1, v) : qx_(u, v) - qx_(u - 1, v);
      const float dqy_y =
          (v == 0) ? qy_(u, v) - qy_(u, v + 1) : qy_(u, v) - qy_(u, v - 1);
      //div_q is AT,forms the negative divergence operator
      const float div_q = dqx_x + dqy_y;
      //
      //g_(u,v) is element-wise weighting matrix
      //
      d_(u, v) = (d_(u, v) + sigmaD_ * (g_(u, v) * div_q + a_(u, v) / theta_))
          / denominator;
    }
  }
}

static float calcEaux(float di,
                      float ai,
                      float costVal,
                      float theta,
                      float lambda)

                      {
  return (0.5f / theta) * ((di - ai) * (di - ai)) + lambda * costVal;
}

void Optimizer::updateA()
{

  const float minInvDepth = costVolume_->getMinInvDepth();
  const float maxInvDepth = costVolume_->getMaxInvDepth();
  const int depthLayers = costVolume_->getDepthLayers();
  const float invDepthStep = (maxInvDepth - minInvDepth) / depthLayers;
  const auto &costs = costVolume_->getCosts();
  const auto &minCost = costVolume_->getMinCost();
  const auto &maxCost = costVolume_->getMaxCost();

  for (int u = 0; u < width_; u++)
  {
    for (int v = 0; v < height_; v++)
    {//for each pixel
      setLambda(u, v);
      const float di = d_(u, v);

      //in paper 2.24
      //here we use the maxcost and the mincost(matrix)
      const float
          r = 2 * theta_ * lambda_ * (maxCost(u, v) - minCost(u, v));

      float EauxMin = 1e+30;
      const float dStart = di - r;
      const float dEnd = di + r;
     //in paper 2.24




      //Largest integer not greater than X. floorf
      const int startLayer =
          max(static_cast<int>(
                  lrintf(floorf((dStart - minInvDepth) / invDepthStep)) - 1),
              0);
      //Smallest integral value not less than X.  ceilf
      const int endLayer =
          min(static_cast<int>(
                  lrintf(ceilf((dEnd - minInvDepth) / invDepthStep)) + 1),
              depthLayers - 1);

      assert(startLayer <= endLayer);

      int minCostLayer = startLayer;
      for (int l = startLayer; l <= endLayer; l++)
      {//ai 就是di
        const float ai = minInvDepth + l * invDepthStep;
        const float Eaux = calcEaux(di, ai, costs[l](u, v), theta_, lambda_);
        if (Eaux < EauxMin)
        {
          EauxMin = Eaux;
          minCostLayer = l;
        }
      }
     //
      a_(u, v) = minInvDepth + float(minCostLayer) * invDepthStep;

    }
  }

}

}