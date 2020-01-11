#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>

using namespace std;
using namespace Eigen;
using namespace cv;

void showPointCloud(
        const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud);

int main()
{
 string tem_path="/home/lzw/2_ear_DTAM/cmake-build-debug/system/result/depth/1.yaml";
 string another_path="/home/lzw/yaml_data/ear_result/result_stereo_in_left_ref_more_depth_layer/176.yaml";
    //
    double fx = 714.28, fy = 714.28, cx = 380.0, cy = 380.0;
    // 基线
    int stero_id=125;string yaml_stereo_path="/home/lzw/yaml_data/ear_result/stereo results/depth/";
     stringstream s_stereo_id;s_stereo_id<<stero_id;
    yaml_stereo_path+=s_stereo_id.str()+".yaml";

      int mono_id=30;stringstream s_mono_id;s_mono_id<<mono_id;
   string yaml_path="/home/lzw/yaml_data/ear_result/depth2/";
     yaml_path+=s_mono_id.str()+".yaml";
   FileStorage fs;
   fs.open(tem_path,FileStorage::READ);
   Mat img;
   fs["img"]>>img;

   Mat depth_img(img.rows,img.cols,CV_32F);
Mat nor_img;
    for (int v = 0; v < img.rows; v++)
    {

        for (int u =0; u < img.cols; u++)
        { depth_img.at<float>(v,u)=1.000/img.at<float>(v,u);


        }
    }
   cv::normalize(depth_img,nor_img,1,0,cv::NORM_MINMAX);

    imshow("inv depth img",img);
    imshow("nor depth img ",nor_img);waitKey(0);
    vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;

    for (int v = 170; v < img.rows-100; v++)//restrict u and v to a small value,can improve vis effect
    {
        for (int u = 190; u < img.cols-70; u++)
        {
            Vector4d point(0, 0, 0, 0.5); // 前三维为xyz,第四维为颜色

            // 根据双目模型计算 point 的位置
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;
            double depth = 1.00000 / (img.at<float>(v, u));
            point[0] = x * depth;
            point[1] = y * depth;
            point[2] = depth;
           if(depth<10||depth>80){continue;} //delete some obviously wrong depth
           if(u<200&&v<220){ continue;}
            pointcloud.push_back(point);
        }
    }
    //cv::imshow("disparity", disparity / 96.0);
    cv::waitKey(0);
    // 画出点云
    showPointCloud(pointcloud);
    return 0;
}

void showPointCloud(
        const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud)
        {

    if (pointcloud.empty())
    {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud)
        {
            glColor3f(255,0,170);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}