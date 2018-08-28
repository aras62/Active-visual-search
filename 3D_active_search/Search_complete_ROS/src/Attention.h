/*
 * Attention.h
 *
 *      Author: Amir Rasouli
 *      email: aras@eecs.yorku.ca
 */

#ifndef ATTENTION_H_
#define ATTENTION_H_

//#include "Control.h"
#include <fstream>
#include <sys/stat.h>
#include <stdio.h>
#include <sstream>
#include <string>
#include <iostream>

// ROS libraries
#include "ros/ros.h"
#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float32.h>
#include <sensor_msgs/fill_image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Bool.h>
#include <ros/callback_queue.h>
#include <ros/subscribe_options.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float64.h>
#include <cv_bridge/cv_bridge.h>
#include <boost/algorithm/string.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"

// Libraries for saliency implemented using ROS wrapper
#include <saliency/GetBackProj.h> // include if use ros package
#include <saliency/GetAIM.h> // include if use ros package
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#define UNKNOWN_SPACE_FLAG -1
#define PIXEL_WISE 1
#define CHANNEL_WISE 2
#define OPPONENT_AXIS 3
#define COMPREHENSIVE 4
#define PI 3.14159265359

enum colorSpace {RGB=0, HSV, Lab, Luv, HSI, HSL, CMY, C1C2C3, COPP, YCrCb, YIQ, XYZ, UVW, YUV,
    OPP, NOPP, xyY, rg, YES, I1I2I3};

class Attention
{
public:
	Attention();
	virtual ~Attention();

	//****************************** Utilities ******************************
	cv::Mat imageConversion(cv::Mat inputImg,colorSpace type, bool norm = true);
	cv::Mat normalizeImage(cv::Mat RGBImage ,int method = PIXEL_WISE);
	void normalizeHistogram(cv::Mat &histogram);
	cv::Mat percentileThreshold(cv::Mat salMap, double percentile);
	std::vector<std::string> getFilesAndDirectories(std::string path, std::vector<std::string> &files);
	void rotation2D(double &y, double &x, double angle);
	sensor_msgs::Image fillImageMsgs(cv::Mat image, std::string imgName);
	cv::Mat getImageFromMsg(sensor_msgs::Image msg);
	//****************************** Methods ******************************
	cv::Mat getBackProj(cv::Mat imageInput, cv::Mat temp, std::string cSpace, bool normal = false, int bins = 64, bool thresh = true);
	cv::Mat getAIM(cv::Mat imageInput, float percent, float scale_factor = 1,
			std::string basisName = "../21infomax950.bin");
	void loadBasis(std::string filename);
	cv::Mat runAIM();

	//*********************************** ROS Version **********************************************
	bool getAIMROS(cv::Mat inputImg, cv::Mat &infoMap, float percent = 0.f,
			float scaleFac = 1,	std::string base_name = "src/saliency/21infomax950.bin");

	bool getBackProjROS(cv::Mat inputImg, cv::Mat tempImg, std::string cSpace,
			cv::Mat &bpImage, bool normal = false, int bins = 64);


public:
	static Attention*_instance;
	std::vector<std::string> _colors;
	cv::Mat adj_sm;
private:
	cv::Mat **kernels;
	float* data;
	float scale, percentile;
	std::vector<cv::Mat> channels;
	cv::Mat image, temp, sm, hist;
	cv::Mat *aim_temp;
	int num_kernels, kernel_size, num_channels;
	std::unique_ptr<ros::NodeHandle> rosNode;
	std::string _namespace;
	int counter;
	double maxVal, minVal, max_aim, min_aim;

};


#endif /* ATTENTION_H_ */
