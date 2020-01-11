//
// Created by 宋孝成 on 2019-04-30.
//

#ifndef DTAM_OPTIMIZER_HPP
#define DTAM_OPTIMIZER_HPP

#include <Camera.hpp>
#include <Frame.hpp>
#include "utils/MatrixType.hpp"

namespace dtam {

class Optimizer {

 public:

  using Ptr = std::shared_ptr<Optimizer>;

  static Optimizer::Ptr create(const Camera::Ptr &camera,
                               const CostVolume::Ptr &costVolume,
                               float alphaG,
                               float betaG,
                               float epsilon,
                               float lambda,
                               float thetaBegin,
                               float thetaEnd) {
    return std::make_shared<Optimizer>(camera,
                                        costVolume,
                                        alphaG,
                                        betaG,
                                        epsilon,
                                        lambda,
                                        thetaBegin,
                                        thetaEnd);
  }

  Optimizer(const Camera::Ptr &camera,
            CostVolume::Ptr costVolume,
            float alphaG,
            float betaG,
            float epsilon,
            float lambda,
            float thetaBegin,
            float thetaEnd) :
      width_(camera->getWidth()),
      height_(camera->getHeight()),
      costVolume_(std::move(costVolume)),
      alphaG_(alphaG),
      betaG_(betaG),
      epsilon_(epsilon),
      lambda_(lambda),
      thetaBegin_(thetaBegin),
      thetaEnd_(thetaEnd) {}

  void initByReferenceFrame(const Frame::Ptr &referenceFrame);

  void optimize(int nIters);


  //question: so the  matrix a_ is what we want?
  const MatrixXXf &getInvDepth() const { return a_; }

  void setLambda() { lambda_ = 1/(1+0.5/a_.maxCoeff()); }
  //in paper 2.2.6,lambda
  void setLambda(int u, int v) { lambda_ = 1/(1+0.5/a_(u,v)); }


 protected:

  MatrixXXf d_;
  MatrixXXf qx_;
  MatrixXXf qy_;//qx_,qy_？？？
  MatrixXXf g_;
  MatrixXXf a_;

  const int width_, height_;
  const CostVolume::Ptr costVolume_;

  const float alphaG_, betaG_;
  const float epsilon_;
  const float thetaBegin_, thetaEnd_;

  float lambda_;
  float theta_ = 0.0f;
  float sigmaD_ = 0.0f, sigmaQ_ = 0.0f;
  float L_ = 2.0f;

  void updateQ();
  void updateD();
  void updateA();
  void computeSigmas();
  void computeG(const Frame::Ptr &referenceFrame);
  void computeL();

};

}
#endif //DTAM_OPTIMIZER_HPP
