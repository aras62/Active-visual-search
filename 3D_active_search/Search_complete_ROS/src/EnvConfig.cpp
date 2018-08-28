/*
 *      Author: Amir Rasouli
 *      email: aras@eecs.yorku.ca
 *
 *		This code is for setting up all hardware parameters
 *
 * 	    Copyright 2017 Amir Rasouli
 *      Licensed under the Simplified BSD License
 */

#include "EnvConfig.h"

using namespace cv;
using namespace std;

//TODO  Set the pant-tilt unit's parameters
PanTiltConfig::PanTiltConfig()
{
	maxTilt = 30;
	minTilt = -20;
	maxPan = 158;
	panDelta = 35; // The resolution of the pan angle changes. min can be 1 degree and max the max angle
	tiltDelta = 20; // The resolution of the tilt angle changes. min can be 1 degree and max the max angle
	pan = 0.f; // Starting pan angle
	tilt = 0.f;// Starting tilt angle
}

//TODO  Set the camera's parameters
CameraConfig::CameraConfig()
{
	cameraHeight = 720;//mm  // How far the camera is from the ground
	cameraHorizontalViewAngle = 66.f;
	cameraVerticalViewAngle = 50.f;
	baseLine = 119.548f;//mm  // Stereo camera baseline
    focalLength = 500.470398f;//pixel
    cameraToRobotFrontDistance = 430;//mm  // The camera distance to the surface of the platform
    cameraToTiltPointDistance = 50;//mm  // The distance of the camera to the center of the pan-tilt unit
    cameraEffectiveRange = 5000.f;//mm  // This parameter defines the effective range of the stereo camera for estimating depth
}

//TODO  Set the robot's parameters
RobotConfig::RobotConfig()
{
robotSpeed = 350;//max speed 700 mm/sec
robotRadius = 350;//mm
robotLength = 650;//mm
robotWidth = 500;//mm
}
EnvConfig::EnvConfig() {
	initWithDefaults();}
EnvConfig::~EnvConfig() {
	// TODO Auto-generated destructor stub
}

void EnvConfig::initWithDefaults()
{
	envSize= Point3i(4000,3000,1000);  // Dimensions of the environment
	voxelSize = 10; // Resolution of the search environment
	robotPos = Point2d(0.,0.); // the initial position of the robot
	robotDir = Point2f(1.,0.); // The initial direction of the robot

	// The effective range of the recognition algorithm for detecting the object
	// These parameters depend on the resolution of the camera, the recognition algorithm, and the nature of the object
	recognitionMaxRadius = 3000.f;
	recognitionMinRadius= 500.f;

	searchMethod = SearchConfig::lookMove; // The seach methods, greedy, lookMove, lookMoveUnknown
	searchThreshold = 0.03; // The search threshold for lookMove methods
}
