#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
using namespace std;
using namespace cv;
int check_hereshold=3;
Mat global_K = (Mat_<double>(3, 3) << 714.28, 0, 380.0, 0, 714.28, 380.0, 0, 0, 1);
struct T
{
    Mat mR,mt,mT;
};
void stereo_depth(vector<KeyPoint> & keypoint_1,
                  vector<KeyPoint> & keypoint_2,
                  vector<DMatch>  & matches,
                  Point3d &test_point3d);
void stereo_depth_right(vector<KeyPoint> & keypoint_1,
                        vector<KeyPoint> & keypoint_2,
                        vector<DMatch>  & matches,
                        Point3d &test_point3d);
Point2d pixel2cam(const Point2d &p, const Mat &K);
void test_for_stereo_t();
void test_for_orb_T();
void test_for_orb_stereo();
int main()
{
   test_for_orb_stereo();

    return 0;
}
void check_row(  vector <KeyPoint>& RAN_KP1,
                 vector<KeyPoint>& RAN_KP2,
                 vector <DMatch> &RR_matches);
void test_for_stereo_t()
{
    Mat img_1=imread("/home/lzw/project_data/for_orb/for_orb_2/image_0/000009.png");
    Mat img_2=imread("/home/lzw/project_data/for_orb/for_orb_2/image_1/000009.png");

    std::vector<KeyPoint> vkl;
    std::vector<KeyPoint> vkr;
    std::vector<DMatch> v_matches;

    Ptr<Feature2D> f2d =ORB::create( 200, 1.2f,8, 31,
            0, 2, ORB::HARRIS_SCORE, 31, 20);
    if (!img_1.data || !img_2.data)
    {
        cerr << "Reading picture error！" << endl;
    }

    f2d->detect(img_1, vkl);
    f2d->detect(img_2, vkr);
    Mat descriptors_1, descriptors_2;
    f2d->compute(img_1, vkl, descriptors_1);
    f2d->compute(img_2, vkr, descriptors_2);
    Mat img_keypoints_1, img_keypoints_2;

    Ptr<DescriptorMatcher> p_matcher = DescriptorMatcher::create("BruteForce-Hamming");

    p_matcher->match(descriptors_1, descriptors_2, v_matches);  Mat img_matches;


    double min_dist = v_matches[0].distance, max_dist = v_matches[0].distance;
    for (int m = 0; m < v_matches.size(); m++)
    {
        if (v_matches[m].distance<min_dist)
        {
            min_dist = v_matches[m].distance;
        }
        if (v_matches[m].distance>max_dist)
        {
            max_dist = v_matches[m].distance;
        }
    }
    vector<DMatch> v_goodMatches;
    for (int m = 0; m < v_matches.size(); m++)
    {
        if (v_matches[m].distance < 0.7*max_dist)
        {
            v_goodMatches.push_back(v_matches[m]);
        }
    }

    Mat good_match_image;
    drawMatches(img_1, vkl, img_2, vkr, v_goodMatches, good_match_image);
    imshow("good_match_image",good_match_image);waitKey(0);
     vector<DMatch> v_test_match;
   for(int i=0;i<v_goodMatches.size();i++)
   {   if(vkl[v_goodMatches[i].queryIdx].pt.y<300)
       {
           v_test_match.push_back(v_goodMatches[i]);
       }

   }
    Mat test_match_image;
    drawMatches(img_1, vkl, img_2, vkr, v_test_match, test_match_image);
    imshow("test_match_image",test_match_image);waitKey(0);


    vector<DMatch> v_single_match;v_single_match.push_back(v_test_match[0]);

    Mat single_match_image;
    drawMatches(img_1, vkl, img_2, vkr, v_single_match, single_match_image);
    imshow("single_match_image",single_match_image);waitKey(0);
    cout<<"K"<<endl<<global_K<<endl;
    cout<<"left pixel: "<<vkl[v_single_match[0].queryIdx].pt<<endl;
    cout<<"right pixel: "<<vkr[v_single_match[0].trainIdx].pt<<endl;
     Point3d p_c1;
    stereo_depth(vkl,vkr,v_single_match,p_c1);
    cout<<"point in left cam: "<<setprecision(10)<<p_c1<<endl;
    Point3d stereo_t(1.8,0,0);
    p_c1-=stereo_t;
    cout<<"point in right cam: "<<setprecision(10)<<p_c1<<endl;
     p_c1/=p_c1.z;
     cout<<" normalized point in right cam: "<<setprecision(10)<<p_c1<<endl;
     double re_u,re_v;
     re_u=global_K.at<double>(0,0)*p_c1.x+global_K.at<double>(0,2)*p_c1.z;
    re_v=global_K.at<double>(1,1)*p_c1.y+global_K.at<double>(1,2)*p_c1.z;
    Point2d repro_pixel(re_u,re_v);Scalar color(0,0,255);

    circle(img_2,repro_pixel,1,color);imshow("repro",img_2);waitKey(0);
cout<<"reproject pixel: "<<setprecision(12)<<re_u<<" "<<re_v<<endl;




}

void test_for_orb_stereo()
{   int id=145;
    stringstream left_path_part,right_path_part;
    left_path_part << setfill('0') << setw(6) << id;



    Mat img_1=imread("/home/lzw/for_orb_2/image_0/"+left_path_part.str()+".png");
    Mat img_2=imread("/home/lzw/for_orb_2/image_1/"+left_path_part.str()+".png");
    Mat ref_left_img=imread("/home/lzw/for_orb_2/image_0/000000.png");
    Mat ref_right_img=imread("/home/lzw/for_orb_2/image_1/000000.png");


    std::vector<KeyPoint> vkl;
    std::vector<KeyPoint> vkr;
    std::vector<DMatch> v_matches;

    Ptr<Feature2D> f2d =ORB::create( 200, 1.2f,8, 31,
                                     0, 2, ORB::HARRIS_SCORE, 31, 20);
    if (!img_1.data || !img_2.data)
    {
        cerr << "Reading picture error！" << endl;
    }

    f2d->detect(img_1, vkl);
    f2d->detect(img_2, vkr);
    Mat descriptors_1, descriptors_2;
    f2d->compute(img_1, vkl, descriptors_1);
    f2d->compute(img_2, vkr, descriptors_2);
    Mat img_keypoints_1, img_keypoints_2;

    Ptr<DescriptorMatcher> p_matcher = DescriptorMatcher::create("BruteForce-Hamming");

    p_matcher->match(descriptors_1, descriptors_2, v_matches);  Mat img_matches;


    double min_dist = v_matches[0].distance, max_dist = v_matches[0].distance;
    for (int m = 0; m < v_matches.size(); m++)
    {
        if (v_matches[m].distance<min_dist)
        {
            min_dist = v_matches[m].distance;
        }
        if (v_matches[m].distance>max_dist)
        {
            max_dist = v_matches[m].distance;
        }
    }
    vector<DMatch> v_goodMatches;
    for (int m = 0; m < v_matches.size(); m++)
    {
        if (v_matches[m].distance < 0.7*max_dist)
        {
            v_goodMatches.push_back(v_matches[m]);
        }
    }
    check_row(vkl,vkr,v_goodMatches);
    Mat good_match_image;
    drawMatches(img_1, vkl, img_2, vkr, v_goodMatches, good_match_image);
    imshow("good_match_image",good_match_image);waitKey(0);
    vector<DMatch> v_test_match;
    for(int i=0;i<v_goodMatches.size();i++)
    {   if(vkl[v_goodMatches[i].queryIdx].pt.y<400)
        {
            v_test_match.push_back(v_goodMatches[i]);
        }

    }
    Mat test_match_image;
    drawMatches(img_1, vkl, img_2, vkr, v_test_match, test_match_image);
    imshow("test_match_image",test_match_image);waitKey(0);
    if(v_test_match.size()!=1){cerr<<"there are more than one match!!"<<endl;}

    vector<DMatch> v_single_match;v_single_match.push_back(v_test_match[0]);

    Mat single_match_image;
    drawMatches(img_1, vkl, img_2, vkr, v_single_match, single_match_image);
    imshow("single_match_image",single_match_image);waitKey(0);
    cout<<"K"<<endl<<global_K<<endl;
    cout<<"left pixel: "<<vkl[v_single_match[0].queryIdx].pt<<endl;
    cout<<"right pixel: "<<vkr[v_single_match[0].trainIdx].pt<<endl;

    Point3d p_c2;
    stereo_depth_right(vkl,vkr,v_single_match,p_c2);
    cout<<"point in right cam: "<<setprecision(10)<<p_c2<<endl;
    Point3d stereo_t(1.8,0,0);

    string trajectory_file="/home/lzw/orb_data/CameraTrajectory.txt";
    ifstream fin(trajectory_file);
    if (!fin) {
        cerr << "cannot find trajectory file at " << trajectory_file << endl;

    }
    vector<T> v_poses;
    while (!fin.eof())
    {   T T;
        double  a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4;
        fin >>  a1>>a2>>a3>>a4>>b1>>b2>>b3>>b4>>c1>>c2>>c3>>c4;
        T.mT=(Mat_<double>(4,4) << a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4,0,0,0,1);
        T.mR= (Mat_<double>(3,3) << a1,a2,a3,b1,b2,b3,c1,c2,c3);
        T.mt= (Mat_<double>(3,1) << a4,b4,c4);


        v_poses.push_back(T);
        //cout<<"R: "<<T.mR<<endl<<"t: "<<T.mt<<endl;

    }
    // cout<<v_poses[0].mT<<endl<<v_poses[1].mT<<endl;
    Mat p_ref_left=(Mat_<double>(4,1)<<0,0,0,1);
    Mat p_ref_right=(Mat_<double>(4,1)<<0,0,0,1);
    Mat mat_p_c2=(Mat_<double>(4,1)<<p_c2.x,p_c2.y,p_c2.z,1);
    Mat right_left = (Mat_<double>(4,4) << 1,0,0,1.8,
                                                    0,1,0,0,
            0,0,1,0,
            0,0,0,1);
    Mat left_right = (Mat_<double>(4,4) << 1,0,0,-1.8,
                                             0,1,0,0,
                                                 0,0,1,0,
                                                  0,0,0,1);



    p_ref_right=left_right*v_poses[id].mT*right_left*mat_p_c2;
    cout<<"p_ref_right:  "<<p_ref_left<<endl;


    p_ref_right/=p_ref_right.at<double>(2,0);
    cout<<" normalized point in right cam: "<<setprecision(10)<<p_ref_right<<endl;
    double re_u,re_v;
    re_u=global_K.at<double>(0,0)*p_ref_right.at<double>(0,0)+global_K.at<double>(0,2)*p_ref_right.at<double>(2,0);
    re_v=global_K.at<double>(1,1)*p_ref_right.at<double>(1,0)+global_K.at<double>(1,2)*p_ref_right.at<double>(2,0);
    Point2d repro_pixel(re_u,re_v);Scalar color(0,0,255);

    circle(ref_right_img,repro_pixel,5,color);
    imshow("ref_left_img",ref_right_img);waitKey(0);
    cout<<"reproject pixel: "<<setprecision(12)<<re_u<<" "<<re_v<<endl;




}

void test_for_orb_T()
{   int id=19;
  stringstream left_path_part,right_path_part;
  left_path_part << setfill('0') << setw(6) << id;



    Mat img_1=imread("/home/lzw/for_orb_2/image_0/"+left_path_part.str()+".png");
    Mat img_2=imread("/home/lzw/for_orb_2/image_1/"+left_path_part.str()+".png");
    Mat ref_left_img=imread("/home/lzw/for_orb_2/image_0/000000.png");
    Mat ref_right_img=imread("/home/lzw/for_orb_2/image_0/000000.png");


    std::vector<KeyPoint> vkl;
    std::vector<KeyPoint> vkr;
    std::vector<DMatch> v_matches;

    Ptr<Feature2D> f2d =ORB::create( 200, 1.2f,8, 31,
                                     0, 2, ORB::HARRIS_SCORE, 31, 20);
    if (!img_1.data || !img_2.data)
    {
        cerr << "Reading picture error！" << endl;
    }

    f2d->detect(img_1, vkl);
    f2d->detect(img_2, vkr);
    Mat descriptors_1, descriptors_2;
    f2d->compute(img_1, vkl, descriptors_1);
    f2d->compute(img_2, vkr, descriptors_2);
    Mat img_keypoints_1, img_keypoints_2;

    Ptr<DescriptorMatcher> p_matcher = DescriptorMatcher::create("BruteForce-Hamming");

    p_matcher->match(descriptors_1, descriptors_2, v_matches);  Mat img_matches;


    double min_dist = v_matches[0].distance, max_dist = v_matches[0].distance;
    for (int m = 0; m < v_matches.size(); m++)
    {
        if (v_matches[m].distance<min_dist)
        {
            min_dist = v_matches[m].distance;
        }
        if (v_matches[m].distance>max_dist)
        {
            max_dist = v_matches[m].distance;
        }
    }
    vector<DMatch> v_goodMatches;
    for (int m = 0; m < v_matches.size(); m++)
    {
        if (v_matches[m].distance < 0.7*max_dist)
        {
            v_goodMatches.push_back(v_matches[m]);
        }
    }
    check_row(vkl,vkr,v_goodMatches);
    Mat good_match_image;
    drawMatches(img_1, vkl, img_2, vkr, v_goodMatches, good_match_image);
    imshow("good_match_image",good_match_image);waitKey(0);
    vector<DMatch> v_test_match;
    for(int i=0;i<v_goodMatches.size();i++)
    {   if(vkl[v_goodMatches[i].queryIdx].pt.y<200)
        {
            v_test_match.push_back(v_goodMatches[i]);
        }

    }
    Mat test_match_image;
    drawMatches(img_1, vkl, img_2, vkr, v_test_match, test_match_image);
    imshow("test_match_image",test_match_image);waitKey(0);
  if(v_test_match.size()!=1){cerr<<"there are more than one match!!"<<endl;}

    vector<DMatch> v_single_match;v_single_match.push_back(v_test_match[0]);

    Mat single_match_image;
    drawMatches(img_1, vkl, img_2, vkr, v_single_match, single_match_image);
    imshow("single_match_image",single_match_image);waitKey(0);
    cout<<"K"<<endl<<global_K<<endl;
    cout<<"left pixel: "<<vkl[v_single_match[0].queryIdx].pt<<endl;
    cout<<"right pixel: "<<vkr[v_single_match[0].trainIdx].pt<<endl;
    Point3d p_c1;Point3d p_c2;
    stereo_depth(vkl,vkr,v_single_match,p_c1);
    cout<<"point in left cam: "<<setprecision(10)<<p_c1<<endl;
    Point3d stereo_t(1.8,0,0);

    string trajectory_file="/home/lzw/orb_data/CameraTrajectory.txt";
    ifstream fin(trajectory_file);
    if (!fin) {
        cerr << "cannot find trajectory file at " << trajectory_file << endl;

    }
vector<T> v_poses;
    while (!fin.eof())
    {   T T;
        double  a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4;
        fin >>  a1>>a2>>a3>>a4>>b1>>b2>>b3>>b4>>c1>>c2>>c3>>c4;
        T.mT=(Mat_<double>(3,4) << a1,a2,a3,a4,b1,b2,b3,b4,c1,c2,c3,c4);
        T.mR= (Mat_<double>(3,3) << a1,a2,a3,b1,b2,b3,c1,c2,c3);
        T.mt= (Mat_<double>(3,1) << a4,b4,c4);


        v_poses.push_back(T);
        //cout<<"R: "<<T.mR<<endl<<"t: "<<T.mt<<endl;

    }
   // cout<<v_poses[0].mT<<endl<<v_poses[1].mT<<endl;
Mat p_ref=(Mat_<double>(3,1)<<0,0,0);
    Mat mat_p_c1=(Mat_<double>(3,1)<<p_c1.x,p_c1.y,p_c1.z);

 T test_T=v_poses[id];

    p_ref=test_T.mR*mat_p_c1+test_T.mt;



    p_ref/=p_ref.at<double>(2,0);
    cout<<" normalized point in right cam: "<<setprecision(10)<<p_ref<<endl;
    double re_u,re_v;
    re_u=global_K.at<double>(0,0)*p_ref.at<double>(0,0)+global_K.at<double>(0,2)*p_ref.at<double>(2,0);
    re_v=global_K.at<double>(1,1)*p_ref.at<double>(1,0)+global_K.at<double>(1,2)*p_ref.at<double>(2,0);
    Point2d repro_pixel(re_u,re_v);Scalar color(0,0,255);

    circle(ref_left_img,repro_pixel,5,color);
    imshow("ref_left_img",ref_left_img);waitKey(0);
    cout<<"reproject pixel: "<<setprecision(12)<<re_u<<" "<<re_v<<endl;




}

Point2d pixel2cam(const Point2d &p, const Mat &K)
{
    return Point2d
            (
                    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
            );
}
void stereo_depth(vector<KeyPoint> & keypoint_1,
        vector<KeyPoint> & keypoint_2,
        vector<DMatch>  & matches,
        Point3d &test_point3d)
{
        double x,y,z;
        x=pixel2cam(keypoint_1[matches[0].queryIdx].pt, global_K).x;
        y=pixel2cam(keypoint_1[matches[0].queryIdx].pt, global_K).y;
        //u=keypoint_1[m.queryIdx].pt.x;
        //v=keypoint_2[m.trainIdx].pt.y;
        //baseline=1.8mm
        //因为是视差，所以就更是x轴方向的偏差
        //像素坐标不可能出现负值，但是可以出现小数
        //但是x y 可以出现负值
        z=1.8*global_K.at<double>(0,0)/abs((keypoint_1[matches[0].queryIdx].pt.x-keypoint_2[matches[0].trainIdx].pt.x));
        x=x*z;
        y=y*z;
        test_point3d=Point3_<double>(x,y,z);
}
void stereo_depth_right(vector<KeyPoint> & keypoint_1,
                   vector<KeyPoint> & keypoint_2,
                   vector<DMatch>  & matches,
                   Point3d &test_point3d)
{
    double x,y,z;
    x=pixel2cam(keypoint_2[matches[0].trainIdx].pt, global_K).x;
    y=pixel2cam(keypoint_2[matches[0].trainIdx].pt, global_K).y;
    //u=keypoint_1[m.queryIdx].pt.x;
    //v=keypoint_2[m.trainIdx].pt.y;
    //baseline=1.8mm
    //因为是视差，所以就更是x轴方向的偏差
    //像素坐标不可能出现负值，但是可以出现小数
    //但是x y 可以出现负值
    z=1.8*global_K.at<double>(0,0)/abs((keypoint_1[matches[0].queryIdx].pt.x-keypoint_2[matches[0].trainIdx].pt.x));
    x=x*z;
    y=y*z;
    test_point3d=Point3_<double>(x,y,z);
}
void check_row( vector <KeyPoint>& RAN_KP1, vector<KeyPoint>& RAN_KP2,vector <DMatch> &RR_matches)
{
    vector<DMatch> after_check_match ;
    for(size_t i=0; i <RR_matches.size();i++)
    {
        int tem_q;int tem_t;
        tem_q= RR_matches[i].queryIdx ;tem_t=RR_matches[i].trainIdx;
        if(abs( RAN_KP1[tem_q].pt.y-RAN_KP2[tem_t].pt.y) <=check_hereshold )
        {    DMatch tem;
            tem.queryIdx=tem_q;tem.trainIdx=tem_t;
            after_check_match.push_back(tem);//所以理论上  这个里面存的还是 原来 keypoint1和keypoint2的index
        }
    }

    RR_matches=after_check_match;//最后修改这个值 ，把得到的这个新match返还给rr_matches
}