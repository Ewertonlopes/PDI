#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <math.h>

double alfa;
int decay_slider = 50;
int decay_slider_max = 100;

int focus_slider = 50;
int focus_slider_max = 100;

int x_offset_slider = 50;
int x_offset_slider_max = 100;

int y_offset_slider = 50;
int y_offset_slider_max = 100;

cv::Mat image,out;
char TrackbarName[50];

cv::Mat tiltshift(cv::Mat image,float unfocus, int decay, int offsetx, int offsety);
cv::Mat translateImg(cv::Mat &img, int offsetx, int offsety);
void on_trackbar_decay(int, void*);
void on_trackbar_focus(int, void*);
void on_trackbar_xoffset(int, void*);
void on_trackbar_yoffset(int, void*);

int main(int argvc, char** argv) {

  cv::Mat img = cv::imread(argv[1],cv::IMREAD_COLOR);

  img.copyTo(image);

  cv::namedWindow("TiltShift", 1);

  std::sprintf( TrackbarName, "Decay %d", decay_slider_max );
  cv::createTrackbar( TrackbarName, "TiltShift",
                      &decay_slider,
                      decay_slider_max,
                      on_trackbar_decay );
  on_trackbar_decay(decay_slider, NULL);
  
  std::sprintf( TrackbarName, "Focus %d", focus_slider_max );
  cv::createTrackbar( TrackbarName, "TiltShift",
                      &focus_slider,
                      focus_slider_max,
                      on_trackbar_focus );
  on_trackbar_focus(focus_slider, NULL);

  std::sprintf( TrackbarName, "X Offset %d", x_offset_slider_max );
  cv::createTrackbar( TrackbarName, "TiltShift",
                      &x_offset_slider,
                      x_offset_slider_max,
                      on_trackbar_xoffset);
  on_trackbar_xoffset(x_offset_slider, NULL);

  std::sprintf( TrackbarName, "Y Offset %d", y_offset_slider_max );
  cv::createTrackbar( TrackbarName, "TiltShift",
                      &y_offset_slider,
                      y_offset_slider_max,
                      on_trackbar_xoffset);
  on_trackbar_xoffset(y_offset_slider, NULL);

  cv::waitKey(0);
  cv::imwrite("OutputTilt.jpg", out);
  return 0;

}

void on_trackbar_decay(int, void*)
{
  int decay = decay_slider*2 + 1;
  int xoffset = (x_offset_slider - 50) * 10;
  int yoffset = (y_offset_slider - 50) * 10;
  float unfocus = focus_slider*0.005;
  cv::Mat aux = tiltshift(image,unfocus,decay,xoffset,yoffset);
  aux.copyTo(out);
  cv::imshow("TiltShift", aux);
}

void on_trackbar_focus(int, void*)
{
  int decay = decay_slider*2 + 1;
  int xoffset = (x_offset_slider - 50) * 10;
  int yoffset = (y_offset_slider - 50) * 10;
  float unfocus = focus_slider*0.005;
  cv::Mat aux = tiltshift(image,unfocus,decay,xoffset,yoffset);
  aux.copyTo(out);
  cv::imshow("TiltShift", aux);
}

void on_trackbar_xoffset(int, void*)
{
  int decay = decay_slider*2 + 1;
  int xoffset = (x_offset_slider - 50) * 10;
  int yoffset = (y_offset_slider - 50) * 10;
  float unfocus = focus_slider*0.005;
  cv::Mat aux = tiltshift(image,unfocus,decay,xoffset,yoffset);
  aux.copyTo(out);
  cv::imshow("TiltShift", aux);
}

void on_trackbar_yoffset(int, void*)
{
  int decay = decay_slider*2 + 1;
  int xoffset = (x_offset_slider - 50) * 10;
  int yoffset = (y_offset_slider - 50) * 10;
  float unfocus = focus_slider*0.005;
  cv::Mat aux = tiltshift(image,unfocus,decay,xoffset,yoffset);
  aux.copyTo(out);
  cv::imshow("TiltShift", aux);
}

cv::Mat tiltshift(cv::Mat image,float unfocus, int decay, int offsetx, int offsety)
{
  cv::Mat g_image = cv::Mat::zeros(image.rows, image.cols, CV_32FC3);
  cv::Mat f_image = cv::Mat::zeros(image.rows, image.cols, CV_32FC3);
  cv::Mat unf_image = cv::Mat::zeros(image.rows, image.cols, CV_32FC3);
  cv::Mat f_mask = cv::Mat(image.rows, image.cols, CV_32FC3, cv::Scalar(0, 0, 0));
  cv::Mat unf_mask = cv::Mat(image.rows, image.cols, CV_32FC3, cv::Scalar(1, 1, 1));
  cv::Mat outp = cv::Mat::zeros(image.rows, image.cols, CV_32FC3);
  cv::Vec3f v_output, h_output, f_output;

  int l1 = unfocus*image.rows;
  int l2 = (1-unfocus) * image.rows;

  int c1 = unfocus*image.cols;
  int c2 = (1 - unfocus) * image.cols;

  for (int i = 0; i < image.rows; i++) 
  {
    for (int j = 0; j < image.cols; j++) 
    {
      h_output[0] = (tanh((float(i - l1) / decay)) - tanh((float(i - l2) / decay))) / 2;
      h_output[1] = (tanh((float(i - l1) / decay)) - tanh((float(i - l2) / decay))) / 2;
      h_output[2] = (tanh((float(i - l1) / decay)) - tanh((float(i - l2) / decay))) / 2;

      v_output[0] = (tanh((float(j - c1) / decay)) - tanh((float(j - c2) / decay))) / 2;
      v_output[1] = (tanh((float(j - c1) / decay)) - tanh((float(j - c2) / decay))) / 2;
      v_output[2] = (tanh((float(j - c1) / decay)) - tanh((float(j - c2) / decay))) / 2;

      f_output[0] = std::min(h_output[0],v_output[0]);
      f_output[1] = std::min(h_output[1],v_output[1]);
      f_output[2] = std::min(h_output[2],v_output[2]);

      f_mask.at<cv::Vec3f>(i, j) = f_output;
    }
  }

  translateImg(f_mask, offsetx, offsety);

  unf_mask = unf_mask - f_mask;

  image.convertTo(image, CV_32FC3);
  cv::GaussianBlur(image, g_image, cv::Size(15, 15), 0, 0);

  f_image = image.mul(f_mask);
  unf_image = g_image.mul(unf_mask);

  outp = f_image + unf_image;

  image.convertTo(image, CV_8UC3);
  outp.convertTo(outp, CV_8UC3);

  return outp;
}

cv::Mat translateImg(cv::Mat &img, int offsetx, int offsety)
{
    cv::Mat trans_mat = (cv::Mat_<double>(2,3) << 1, 0, offsety, 0, 1, offsetx);
    cv::warpAffine(img,img,trans_mat,img.size());
    return img;
}