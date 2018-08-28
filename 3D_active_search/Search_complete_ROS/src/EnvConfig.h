/*
 * EnvConfig.h
 *
 *      Author: Amir Rasouli
 *      email: aras@eecs.yorku.ca
 */

#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#ifndef ENVCONFIG_H_
#define ENVCONFIG_H_

namespace SearchConfig
{
	enum searchMethod {greedy, lookMove};
}

class PanTiltConfig
		{
	friend class Environment;
		public:
		PanTiltConfig();
		~PanTiltConfig(){};
		float pan;
		float tilt;
		int maxTilt, minTilt;
		int maxPan;
		int tiltDelta, panDelta;
		};
class CameraConfig{
public:
	CameraConfig();
	~CameraConfig(){};
	int cameraHeight;	//camera height in mm
	float cameraVerticalViewAngle;		//degrees
	float cameraHorizontalViewAngle;	//degrees
	float cameraToRobotFrontDistance;
	float cameraToTiltPointDistance;
	float baseLine;//meter
	float focalLength;//pixel
	float cameraEffectiveRange;//mm
};
class RobotConfig
{
public:
	RobotConfig();
	~RobotConfig(){};
	double robotSpeed;//max speed 700 mm/sec
	int robotRadius;//mm
	int robotLength;//mm
	int robotWidth ;//mm
};
class EnvConfig {
	friend class Environment;
	public:
	EnvConfig ();
	~EnvConfig();
		cv::Point3i envSize;
		float voxelSize;
		cv::Point2d robotPos;
		cv::Point2d robotDir;	//point in environment where robot is facing
		float recognitionMaxRadius, recognitionMinRadius;
		PanTiltConfig PTConf;
		CameraConfig CamConf;
		RobotConfig RobotConf;
		SearchConfig::searchMethod  searchMethod;
		double searchThreshold;
	private:
		void initWithDefaults();
};

#endif /* ENVCONFIG_H_ */
