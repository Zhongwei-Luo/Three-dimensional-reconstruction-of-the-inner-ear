//
// Created by 宋孝成 on 2019-04-30.
//

#include "DenseMapper.hpp"
#include "cost_volume/CostVolume.hpp"
#include "optimizer/Optimizer.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/operations.hpp>
#include "utils/MatrixType.hpp"

using namespace std;
using namespace cv;

namespace dtam
{


DenseMapper::DenseMapper(Camera::Ptr camera, const cv::FileStorage &settings)
    : camera_(std::move(camera)) {

  costVolume_ = CostVolume::create(
      camera_,
      settings["mapper.costvolume.depth_layers"],
      settings["mapper.costvolume.min_inv_depth"],
      settings["mapper.costvolume.max_inv_depth"]);

  optimizer_ = Optimizer::create(
      camera_,
      costVolume_,
      settings["mapper.regularizer.alpha_G"],
      settings["mapper.regularizer.beta_G"],
      settings["mapper.optimizer.epsilon"],
      settings["mapper.optimizer.lambda"],
      settings["mapper.optimizer.theta_begin"],
      settings["mapper.optimizer.theta_end"]);

  maxFramesPerCostVolume_ = settings["mapper.max_frames_per_costvolume"];
  nIters_ = settings["mapper.n_iters"];

}

void DenseMapper::addFrame(const Frame::Ptr &frame)

{
  if (costVolume_->getFrameLayer() == 0)

  {
    costVolume_->setReferenceFrame(frame);
    cout<<"this is ref frame"<<endl;
    cout<<"its Twc=: "<<frame->getTwc()<<endl;
    cout<<"its Tcw=: "<<frame->getTcw()<<endl;
    optimizer_->initByReferenceFrame(frame);
  }
  else
   {
      cout<<"this is m frame in L(r)"<<endl;
       cout<<"its Twc=: "<<frame->getTwc()<<endl;
       cout<<"its Tcw=: "<<frame->getTcw()<<endl;
       cout<<"will be I: "<<frame->getTwc()*frame->getTcw()<<endl;
    costVolume_->updateCost(frame);
    refInvDepth_ = costVolume_->getMinCostInvDepth();

//question:这里修改成这样是不是正确的呢 要不要注释上一行
    //costVolume_->updateCost("result/depth/296.yaml");
  if (optimizer_) optimizer_->optimize(nIters_);
   //question:scoped_lock  what is the meaning of scoped_lock
   scoped_lock updateLock(updateDepthMutex_);


   //
  refInvDepth_ = optimizer_->getInvDepth();


  }
}

cv::Mat DenseMapper::getRefInvDepthImage() {
  Mat ret;
  //question:what is the meaning of scoped_lock
  scoped_lock updateLock(updateDepthMutex_);
  eigen2cv(refInvDepth_.transpose().eval(), ret);
  return ret;
}

cv::Mat DenseMapper::getRefDepthImage() {
  Mat ret;
  scoped_lock updateLock(updateDepthMutex_);
  eigen2cv(costVolume_->getMinCostInvDepth().cwiseInverse().transpose().eval(),
           ret);
  return ret;
}

const std::vector<MatrixXXf> &DenseMapper::getCosts() const {
  return costVolume_->getCosts();
}

}
