//
// Created by 宋孝成 on 2019-04-30.
//

#include "System.hpp"
#include <utils/ResultWritingUtil.hpp>
#include <iostream>
#include <Frame.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/persistence.hpp>
#include <thread>

using namespace dtam;
using namespace std;

System::System(const std::string &dataFolderPath,
       const std::string &settingsFilePath,
       std::string &trac)
       {
  cv::FileStorage settings(settingsFilePath, cv::FileStorage::READ);
  Camera::Ptr camera = Camera::create(
      settings["camera.width"],
      settings["camera.height"],
      settings["camera.fx"],
      settings["camera.fy"],
      settings["camera.cx"],
      settings["camera.cy"]);

  denseMapper_ = DenseMapper::create(camera, settings);
  sparseTracker_ = SparseTracker::create(camera, ORBBF);
  dataset_ = Dataset::create(dataFolderPath, 501); //
  trajectory_file_path_=trac;
}

void System::start()
{
  dtam();
}

void System::stop()
{
  stopped_ = true;
}

void System::dtam()
{
  cv::namedWindow("rgb", cv::WINDOW_AUTOSIZE);
  cv::namedWindow("depth", cv::WINDOW_AUTOSIZE);



    ifstream fin(trajectory_file_path_);
    if (!fin) {
        cerr << "cannot find trajectory file at " << trajectory_file_path_ << endl;

    }

    while (!fin.eof())
    {   dtam::cv_T T;
        double  a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4;
        fin >>  a1>>a2>>a3>>a4>>b1>>b2>>b3>>b4>>c1>>c2>>c3>>c4;
        T.mT=(cv::Mat_<double>(4,4) << a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4,0,0,0,1);
        T.mR= (cv::Mat_<double>(3,3) << a1,a2,a3,b1,b2,b3,c1,c2,c3);
        T.mt= (cv::Mat_<double>(3,1) << a4,b4,c4);


        v_poses_.push_back(T);
        //cout<<"R: "<<T.mR<<endl<<"t: "<<T.mt<<endl;

    }

  while (dataset_->next())
  {
    cout <<"current index_ in while loop: "<< dataset_->getNumber() << endl;
    double timestamp = dataset_->readTimestamp();
    cv::Mat imageRGB = dataset_->readRGB();
    cv::imshow("rgb", imageRGB);
    cv::waitKey(1);
    Eigen::Matrix4f twc = dataset_->readTwc(v_poses_);
    Frame::Ptr frame = Frame::create(timestamp, imageRGB);
    frame->setTwc(twc);

   cout << "add frames" << endl;
    denseMapper_->addFrame(frame);
    cout << "add frames done" << endl;
cv::Mat img = denseMapper_->getRefInvDepthImage();


      if (img.size().width > 0 && img.size().height > 0)

      {
          cv::Mat normalizedImg;

         /*
          for(int u=0;u<img.rows-1;u++) {
              for (int v = 0; v < img.cols-1; v++) {
                  double t;
                  t = 1.0 / img.at<double>(u, v);
                  cout<<"img.at<double>(u, v);"<<img.at<double>(u, v)<<endl;
                  cout << t << endl;

                  img.at<double>(u, v) = t;
              }
          }
          */
          cv::normalize(img, normalizedImg, 1, 0, cv::NORM_MINMAX);
          cv::imshow("depth", normalizedImg);
          cv::waitKey(1);
          saveDepthData(img, std::to_string(dataset_->getNumber() - 1));
          saveDepthImage(normalizedImg, std::to_string(dataset_->getNumber() - 1));
      }
      cout<<"end one while loop"<<endl;
      cout<<"-----------------------"<<endl;
      cout<<endl;
  }

}

void System::showDepthImage() {
  cv::namedWindow("depth", cv::WINDOW_AUTOSIZE);
  while (!stopped_) {
    if (updateDepthPending_) {
      const auto img = denseMapper_->getRefDepthImage();
      if (img.size().width > 0 && img.size().height > 0) {
        cout << "non-empty mat" << endl;
        cv::imshow("depth", img);
      } else {
        cout << "empty mat" << endl;
      }
      cv::waitKey(1);
      updateDepthPending_ = false;
    }
  }
}
