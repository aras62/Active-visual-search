/*
 * Environment.h
 *      Author: Amir Rasouli
 *      email: aras@eecs.yorku.ca
 */

#ifndef ENVIRONMENT_H_
#define ENVIRONMENT_H_

#include "EnvConfig.h"
#include "Attention.h"
#define UNKNOWN_SPACE_FLAG -1
#define SQR(X) ((X)*(X))

class CameraViewDirection
{
	friend class Environment;
public:
	CameraViewDirection()
	{
		Tilt = Pan = Prob =cost=utility= 0.f;
	}
	CameraViewDirection(float pan, float tilt)
	{
		Pan = pan;
		Tilt = tilt;
		Prob = 0.;
		cost = 0.;
		utility = 0.;
	}
	float Pan,Tilt;	//pan degrees
    double Prob; //probability in that direction
    float cost;
    float utility;
private:
	cv::Mat transform;
};
struct BestPolicy
{
	cv::Point3d p;
	CameraViewDirection direction;
	double util, prob, cost;
	double distance;
};
struct ProbabilityLocations
{
	int id;
	cv::Point3d p;
	std::vector<CameraViewDirection> directions;
	double probVisible; // only for lookmove
};




class Environment {
public:
	Environment();
	Environment(EnvConfig &c);
	virtual ~Environment();


	//Global Functions
	static float getAngleOfVector(cv::Point2d origin, cv::Point2d p);
	static float keepAngleWithin180( float angle )
	{
		double intpart;
		float fractpart = modf(angle, &intpart);
		float res = ((int)intpart)%360 + fractpart;
		if(res < -180)
			res = res + 360;
		else if(res > 180)
			res = res - 360;
		return res;
	}
	void calculateNewPan(float &policyPan, float &disPlacement, float prevDisplace);

	//******* Accessors and Mutators
	// TODO Functions have to be added to navigate the robot and set pan and tilt angles of the camera
	float getRobotDir() const {
		return _robotDir;
	}
	void setRobotDir(cv::Point2d robotDir) {
		_robotDir =getAngleOfVector(_robotPos, robotDir);
	}
	const cv::Point2d& getRobotPos() const {
		return _robotPos;
	}
	void setRobotPos(const cv::Point2d& robotPos) {
		_robotPos = robotPos;
	}
    void getPanTilt(float &pan, float &tilt)
    {
    	pan = _PTConfig.pan;
    	tilt = _PTConfig.tilt;
    }
    void setPan(float pan)
     {
      _PTConfig.pan = pan;
     }
    void setTilt(float tilt)
       {
        _PTConfig.tilt = tilt;
       }
	SearchConfig::searchMethod getMethod(){return method;}

    //search
	std::vector<BestPolicy> chooseBestAction(std::vector<CameraViewDirection> directions);
	std::vector<BestPolicy> chooseBestActionGreedy(std::vector<CameraViewDirection> directions);
	std::vector<BestPolicy> chooseBestActionLookMove(std::vector<CameraViewDirection> directions);
	std::vector<CameraViewDirection>buildListOfViewDirections();
	void updateEnvironment();
	cv::Mat visualizeEnvironment();
	void search();

private:
	//intialization
	void init(EnvConfig c);
	void clearAll();
	//Search
	void  buildLocationList(std::vector<ProbabilityLocations> &pointList);
    std::vector<CameraViewDirection> generatePolicies(cv::Point3d location, std::vector<CameraViewDirection> directions);
    double calculateProbabilityOfViewPoint(CameraViewDirection &dir, cv::Point3d probLocs);
    float estimateCost(float pan, float dirPan, cv::Point3d location);
    BestPolicy chooseBestPolicy(std::vector<ProbabilityLocations> pointList);
    BestPolicy chooseBestPolicy(std::vector<CameraViewDirection> directions);
    double computeTotalProbVisibleFromPoint(ProbabilityLocations &point);
    cv::Mat generateSaliencyMap();
    cv::Mat imageToMap(cv::Mat salMap);
    cv::Mat transformation2D(cv::Mat depthImg);
    cv::Mat clearNanInf(cv::Mat matrix);
    cv::Mat map3dTo2d(cv::Mat map);

public:
    cv::Mat _obstacleMap, _envImage,_saliencyImg;
    Attention* _saliency;
private:
	cv::Mat _environment3D;
	cv::Mat _saliencyMap;
	cv::Mat _envTransform;
	float _voxelSize;
	float _recMaxRange,_recMinRange;
	cv::Point2d _robotPos;	//robot position is kept in mm
	float _robotDir;
	CameraConfig _CamConfig;
	PanTiltConfig _PTConfig;
	RobotConfig _RobConfig;
	bool _firstAttemptUknown;
	static Environment*_instance;
	SearchConfig::searchMethod method;
	double searchThreshold;
	int _envMapSize[3];


};

#endif /* ENVIRONMENT_H_ */
