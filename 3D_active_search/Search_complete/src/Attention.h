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

	//****************************** Methods ******************************
	cv::Mat getBackProj(cv::Mat imageInput, cv::Mat temp, std::string cSpace, bool normal = false, int bins = 64, bool thresh = true);
	cv::Mat getAIM(cv::Mat imageInput, float percent, float scale_factor = 1,
			std::string basisName = "../21infomax950.bin");
	void loadBasis(std::string filename);
	cv::Mat runAIM();

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
	int counter;
	double maxVal, minVal, max_aim, min_aim;

};


#endif /* ATTENTION_H_ */
