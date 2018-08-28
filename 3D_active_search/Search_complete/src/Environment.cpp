/*
 *      Author: Amir Rasouli
 *      email: aras@eecs.yorku.ca
 *
 *		This code is for 3D search in unknown environments. The code relies on saliency information to optimize the search process.
 *		There are two methods of search, lookmove [3,4] and greedy [2]. For the details on each search method please
 *		refer to the corresponding papers.
 *
 *		Note: This code is simulating all robot actions. To run the code on a practical platform, the control functions need to be added
 *
 *      Please if you use the code in your research cite one of the following papers:
 *
 *      [1] A. Rasouli and J. K. Tsotsos, "Integrating Three Mechanisms of Visual Attention for Active Visual Search",
 *         in Proc. 8 The International Symposium on Attention in Cognitive Systems IROS, Oct. 2015
 *      [2] A. Rasouli and J. K. Tsotsos, "Sensor Planning for 3D Visual Search with Task Constraints", in Proc.
 *         Canadian Conference on Computer and Robot Vision, May, 2016.
 *      [3] A. Rasouli and J. K. Tsotsos, "Attention in Autonomous Robotic Visual Search", in Proc. i-SAIRAS,  Jun. 2014
 *      [4] A. Rasouli and J. K. Tsotsos, "Visual Saliency Improves Autonomous Visual Search", in Proc. The Conference on Computer and Robotic Vision, May, 2014
 *
 *
 * 	    Copyright 2017 Amir Rasouli
 *      Licensed under the Simplified BSD License
 */

#include "Environment.h"
using namespace cv;
using namespace std;

//Initialization
Environment::Environment() {
	_voxelSize =0;
	_saliency = new Attention;
}
Environment::~Environment() {
	// TODO Auto-generated destructor stub
}
Environment* Environment::_instance = NULL;
Environment::Environment(EnvConfig &c)
{
	init(c);
}
void Environment::clearAll()
{
	_environment3D.release();
	_obstacleMap.release();
	_voxelSize = 0;
	_robotPos = Point2d(0, 0);
	_robotDir = 0;
}
void Environment::init(EnvConfig c)
{
	clearAll();
	_voxelSize = c.voxelSize;
	_envMapSize[0] = round(((double)c.envSize.y) / _voxelSize); // y dimension, rows
	_envMapSize[1] = round(((double)c.envSize.x) / _voxelSize); // x dimension, columns
	_envMapSize[2] = round(((double)c.envSize.z) / _voxelSize); // z dimension, height

	double prob;
	method = c.searchMethod;
	searchThreshold = c.searchThreshold;

	prob = 1. / ((double)_envMapSize[0] * _envMapSize[1] * _envMapSize[2]);


	_environment3D = Mat(3, _envMapSize, CV_64F, Scalar::all(prob));
	_obstacleMap = Mat(3,_envMapSize, CV_32F, Scalar::all(UNKNOWN_SPACE_FLAG));
	_saliencyMap = Mat(3,_envMapSize, CV_32F, Scalar::all(UNKNOWN_SPACE_FLAG));
	_envTransform = Mat(3, _envMapSize, CV_32FC3, Scalar::all(0));
	setRobotPos(c.robotPos);
	setRobotDir(c.robotDir);
	_CamConfig = c.CamConf;
	_CamConfig.cameraHeight = round((double)c.CamConf.cameraHeight / _voxelSize);
	_PTConfig = c.PTConf;
	_RobConfig = c.RobotConf;
	_recMaxRange = c.recognitionMaxRadius;
	_recMinRange = c.recognitionMinRadius;
	_instance = this;
	_firstAttemptUknown = true;

}
vector<CameraViewDirection> Environment::buildListOfViewDirections()
{
	int absMaxTilt = std::max(abs(_PTConfig.maxTilt), abs(_PTConfig.minTilt));
	vector<CameraViewDirection> res;

	for(int p = 30; p < _PTConfig.maxPan; p += _PTConfig.panDelta)
	{
		for(int t = 0; t < absMaxTilt; t += _PTConfig.tiltDelta)
		{
			if(t <= _PTConfig.maxTilt)
			{
				res.push_back(CameraViewDirection(p, t));
				if(p != 0)
					res.push_back(CameraViewDirection(-p , t));

			}
			if(-t >= _PTConfig.minTilt && t != 0)
			{
				res.push_back(CameraViewDirection(p, -t));
				if(p != 0)
					res.push_back(CameraViewDirection(-p, -t));
			}
		}
	}

	return res;
}

//Search

// Choose the best policies at each time
vector<BestPolicy> Environment::chooseBestAction(vector<CameraViewDirection> directions)
{
	if (method == SearchConfig::lookMove)
	{
		return chooseBestActionLookMove(directions);
	}else
		if(method == SearchConfig::greedy)
		{
			return chooseBestActionGreedy(directions);
		}else
		{
			return chooseBestActionGreedy(directions);
		}
}
vector<BestPolicy> Environment::chooseBestActionGreedy(vector<CameraViewDirection> directions)
{

	vector<ProbabilityLocations> pointList;
	vector<BestPolicy> dir;
	BestPolicy bestDir;
	buildLocationList(pointList);

	//	QtConcurrent::blockingMap(pointList, [this, directions](ProbabilityLocations &data)
	//	{
	//		data.directions = calculateDirectionsProbabilities(data, directions);
	//	});

	for (unsigned int i = 0; i < pointList.size();i++)
	{
		pointList[i].directions = generatePolicies(pointList[i].p,directions);
	}
	bestDir = chooseBestPolicy(pointList);

	//		QFile file(SearchConfiguration::getLogFileName());
	//		file.open(QIODevice::Append | QIODevice::Text);
	//		QTextStream out(&file);

	cout << "*******Best Direction Statistics, Greedy*******\n" <<
			"Prob:    " << bestDir.prob << "\n" <<
			"Position:   " << "(" << bestDir.p.x << "," << bestDir.p.y << ")\n"
			<< "Direction(pan,tilt):    " << "(" << bestDir.direction.Pan << "," << bestDir.direction.Tilt << ")\n"
			<< "Utility Value :    " << bestDir.util << "\n"
			<< "Distance:   " << bestDir.distance << "\n" ;//<<
	//"Time Elapsed to Choose Policy:    " << timer.elapsed() << "\n";
	dir.push_back(bestDir);
	return dir;
}
vector<BestPolicy> Environment::chooseBestActionLookMove(vector<CameraViewDirection> directions)
{
	vector<ProbabilityLocations> pointList;
	vector<BestPolicy> dir;
	BestPolicy bestDir;
	Point2d robPos = getRobotPos();
	directions = generatePolicies(Point3d(robPos.x/_voxelSize,robPos.y/_voxelSize,_CamConfig.cameraHeight/_voxelSize),directions);

	bestDir = chooseBestPolicy(directions);
	if (bestDir.direction.Prob < searchThreshold)
	{

		buildLocationList(pointList);
		ProbabilityLocations nextLocation;
		nextLocation.probVisible= 0.;
		Mat locDisplay(_environment3D.size[0],_environment3D.size[1], CV_32F, Scalar::all(0));
		int gridSize = 2*_RobConfig.robotRadius/_voxelSize;
		for (unsigned int i = 0; i< pointList.size();i++)
		{
			pointList[i].probVisible = computeTotalProbVisibleFromPoint(pointList[i]);
			rectangle(locDisplay, Rect(pointList[i].p.x - gridSize / 2, pointList[i].p.y -
					gridSize / 2, gridSize + 1, gridSize + 1), pointList[i].probVisible, CV_FILLED);
			if(pointList[i].probVisible > nextLocation.probVisible)
			{
				nextLocation = pointList[i];
			}
		}

		bestDir.p.x = nextLocation.p.x*_voxelSize;
		bestDir.p.y = nextLocation.p.y*_voxelSize;
		bestDir.distance = sqrt(SQR(bestDir.p.x - robPos.x) + SQR(bestDir.p.y - robPos.y));
		cout << " Next Location is : " <<  "(" << bestDir.p.x << "," << bestDir.p.y << ")\n";
		resize(locDisplay, locDisplay, Size(640,480));
		normalize(locDisplay, locDisplay, 255, 0, NORM_MINMAX);
		locDisplay.convertTo(locDisplay, CV_8U);
		imshow("nextLocation",locDisplay);
		waitKey(1000);
	}

	//	QFile file(SearchConfiguration::getLogFileName());
	//	file.open(QIODevice::Append | QIODevice::Text);
	//	QTextStream out(&file);
	cout << "*******Best Direction Statistics, Greedy*******\n"
			<< "Prob:    " << bestDir.direction.Prob<< "\n"
			<<"Position:   " << "(" << bestDir.p.x << "," << bestDir.p.y << ")\n"
			<< "Direction(pan,tilt):    " << "(" << bestDir.direction.Pan << "," << bestDir.direction.Tilt << ")\n"
			<< "cost:   "<<bestDir.cost<< "\n"
			<< " Utility Value:    " << bestDir.util << "\n"
			<< "Distance:   " << bestDir.distance << "\n";
	dir.push_back(bestDir);

	return dir;
}
BestPolicy Environment::chooseBestPolicy(vector<ProbabilityLocations> pointList)
{
	BestPolicy bestDir;
	bestDir.util = 0.f;
	Point2d robotPos = getRobotPos();
	double utility;
	for (unsigned int i = 0; i < pointList.size(); i++){
		for (unsigned int j = 0; j < pointList[i].directions.size(); j++){

			utility= pointList[i].directions[j].utility;
			if (utility > bestDir.util)
			{
				bestDir.prob = pointList[i].directions[j].Prob;
				bestDir.p = pointList[i].p *_voxelSize;
				bestDir.direction = pointList[i].directions[j];
				bestDir.cost = pointList[i].directions[j].cost;
				bestDir.util = pointList[i].directions[j].utility;
				bestDir.distance = sqrt(SQR(robotPos.x-bestDir.p.x)+SQR(robotPos.y-bestDir.p.y));
			}
		}
	}
	return bestDir;
}
BestPolicy Environment::chooseBestPolicy(vector<CameraViewDirection> directions)
{
	BestPolicy bestDir;
	bestDir.prob = 0.;
	Point2d robotPos = getRobotPos();
	double probability = 0.;
	for (unsigned int j = 0; j < directions.size(); j++){
		probability = directions[j].Prob;
		if (probability > bestDir.prob)
		{
			bestDir.prob = directions[j].Prob;
			bestDir.p = Point3d(robotPos.x,robotPos.y,_CamConfig.cameraHeight);
			bestDir.direction = directions[j];
			bestDir.cost = directions[j].cost;
			bestDir.util = directions[j].utility;
			bestDir.distance = 0;
		}
	}
	return bestDir;
}

// Builds a list of potential locations the robot can move to
void  Environment::buildLocationList(vector<ProbabilityLocations> &pointList)
{

	int c = 0;
	int gridSize = 4*_RobConfig.robotRadius/_voxelSize;//2*

	float robotRadiusSqr = SQR(gridSize/2);
	Point2d robotLocalCord = Point2d(_robotPos.x/_voxelSize,_robotPos.y/_voxelSize);
	float cameraHeight = _CamConfig.cameraHeight/_voxelSize;
	///Current Position of the Robot to add to the list

	ProbabilityLocations t;
	t.id = c++;
	t.p = Point3d(robotLocalCord .x, robotLocalCord .y, cameraHeight);
	pointList.push_back(t);


	//****************Calculate Locations*********************
	for (int j = gridSize / 2.; j < _environment3D.size[0] - (_environment3D.size[0] % gridSize); j = j + gridSize) // for(int j = 0; j < _obstacleMap.rows; j++)
	{

		for (int i = gridSize / 2.; i < _environment3D.size[1]- (_environment3D.size[1] % gridSize); i = i + gridSize) //for(int i = 0; i < _obstacleMap.cols; i++)
		{
			float distSqr = SQR(i - robotLocalCord .x) + SQR(j - robotLocalCord .y);
			float location =  _obstacleMap.ptr<schar>(j)[i];
			if (location <= 0 &&	//point is not inside an obstacle
					distSqr > robotRadiusSqr)
			{

				ProbabilityLocations t;
				t.id = c++;
				t.p = Point3d(i , j , cameraHeight);
				pointList.push_back(t);
			}
		}
	}

}
vector<CameraViewDirection> Environment::generatePolicies(Point3d location, vector<CameraViewDirection> directions)
{
	float pan = 0.f;
	float junk = 0.f;
	getPanTilt(pan, junk);
	for (unsigned int i = 0; i < directions.size(); i++)
	{
		directions[i].Prob = calculateProbabilityOfViewPoint(directions[i], location);
		directions[i].cost = estimateCost(pan, directions[i].Pan, location);
		directions[i].utility = (directions[i].cost == 0) ? directions[i].Prob : directions[i].Prob / directions[i].cost;

	}

	return directions;
}
double Environment::calculateProbabilityOfViewPoint(CameraViewDirection &dir, Point3d probLocs)
{
	const double HFOV = _CamConfig.cameraHorizontalViewAngle/ 2.;
	float recMaxRange =  _recMaxRange/ _voxelSize;
	float recMinRange =  _recMinRange/_voxelSize;
	float robotDir = getRobotDir();
	double res = 0.;
	float corrPan = dir.Pan + robotDir;

	for (int i = 0; i < _environment3D.size[1]; i++)
	{
		for (int j = 0; j < _environment3D.size[0]; j++)
		{
			for (int k = 0; k < _environment3D.size[2]; k++)
			{
				Point3d p(i, j, k);
				Point3d vec = p - probLocs;
				double length = sqrt(SQR(vec.x) + SQR(vec.y)+SQR(vec.z));
				float angV = acos(vec.z/length)*180/PI;
				float angH= getAngleOfVector(Point2d(probLocs.x,probLocs.y), Point2d(p.x,p.y));

				if (	  (fabs(corrPan - angH) >HFOV
						&& fabs(corrPan + 360 - angH) > HFOV
						&& fabs(corrPan - 360 - angH) > HFOV)
						||
						fabs((90-angV)-dir.Tilt) >HFOV)

				{
					break;
				}
				else if (length < recMaxRange && length > recMinRange)
				{
					res += _environment3D.ptr<double>(j)[i*_environment3D.size[2]+k];
				}
			}
		}
	}
	return res;
}

// Estimates the cost of each policy
float Environment::estimateCost(float pan, float dirPan, Point3d location)
{
	Point2d robotPos = getRobotPos();
	float dist = sqrt(SQR(robotPos.x / _voxelSize - location.x) + SQR(robotPos.y / _voxelSize - location.y))*_voxelSize;
	double panTiltTime = fabs(dirPan - pan)*0.027;
	double travelTime = (dist<=_CamConfig.cameraEffectiveRange) ?///_voxelSize
			dist/_RobConfig.robotSpeed  :
			(dist-_CamConfig.cameraEffectiveRange)*dist/_RobConfig.robotSpeed;
	return panTiltTime+travelTime;
}
double Environment::computeTotalProbVisibleFromPoint(ProbabilityLocations &point)
{
	double radius = _recMaxRange/ _voxelSize;
	double maxRangeSquare = SQR(radius);
	float minRangeSquare = SQR(_recMinRange / _voxelSize);
	double res = 0.;
	int xStart, xEnd, yStart, yEnd;
	xStart = ((point.p.x - radius) < 0) ? 0 : point.p.x - radius;
	xEnd = ((point.p.x + radius) > _environment3D.size[1]) ? _environment3D.size[1] : point.p.x + radius;
	yStart = ((point.p.y - radius) < 0) ? 0 : point.p.y - radius;
	yEnd = ((point.p.y + radius) > _environment3D.size[0]) ? _environment3D.size[0] : point.p.y + radius;

	Point2i r;
	r.x = point.p.x;
	r.y = point.p.y;


	for (int i = xStart; i < xEnd; i++)//_enviroment3D.size[1]
	{
		for (int j = yStart; j < yEnd; j++)//_enviroment3D.size[0]
		{
			for (int k = 0; k < _environment3D.size[2]; k++)
			{
				Point2i p(i, j);
				Point2f vec = p - r;
				double lengthSqr = (SQR(vec.x) + SQR(vec.y));
				if (lengthSqr < maxRangeSquare && lengthSqr > minRangeSquare)
				{
					res += _environment3D.ptr<double>(j)[i*_environment3D.size[2]+k];
				}

			}
		}
	}
	return res;
}

//Global Functions
float Environment::getAngleOfVector(Point2d origin, Point2d p)
{
	Point2d v = p - origin;
	if(v.x == 0 && v.y == 0)
		return 0.f;
	float angle = fastAtan2(-v.y, v.x);
	return angle;
}
cv::Mat Environment::imageToMap(cv::Mat salMap)
{
	//TODO: Complete this function to read the depthmap from the robot and tranform salmap
	Mat depthImage; // TODO Get the 3 channel depth map from the sensor
	Mat map = Mat(3,_envMapSize, CV_64F, Scalar::all(0));
	Mat depthMap = transformation2D(depthImage); //channel 0 z (depth), channel 1 x

	for (int r = 0; r < depthImage.rows; r++)
		for (int c = 0; c < depthImage.cols; c++)
		{
			if (isnan(depthImage.ptr<float>(r)[c*3+2]))
				continue;

			int x = depthMap.ptr<float>(r)[c*4]/_voxelSize + _envMapSize[1]/2;
			int y = _envMapSize[0] -(depthMap.ptr<float>(r)[c*4+1]/_voxelSize + _envMapSize[0]/2);
			int z = depthMap.ptr<float>(r)[c*4+2]/_voxelSize;
			if( x > 0 && x < _envMapSize[1] && y > 0 && y <_envMapSize[0] && z > 0 && z <_envMapSize[2])
			{
				map.ptr<double>(y)[x*_envMapSize[2]+z] += salMap.ptr<double>(r)[c];

			}
		}
	return map;
}
cv::Mat Environment::transformation2D(cv::Mat depthImg)
{
	float pan, tilt;
	getPanTilt(pan, tilt);

	Point2d robPose =  getRobotPos(); // Has to be changed to actual position of the robot
	double phi =   getRobotDir() + pan; // Change to the actual direction of the robot


	Mat depthVec = depthImg.clone();
	depthVec = depthVec.reshape(3,1);

	vector<Mat> channels;
	split(depthVec, channels);
	Mat ones = Mat(channels[0].size(), CV_32F, Scalar::all(1));


	Mat merge;
	vconcat(channels[2], -channels[0], merge);
	vconcat(merge, channels[1], merge);
	vconcat(merge, ones, merge);


	Mat R =  (Mat_<float>(4, 4) <<
			cos(phi), -sin(phi), 0, robPose.x,
			sin(phi), cos(phi), 0, robPose.y,
			0, 0, 1, _CamConfig.cameraHeight,
			0, 0, 0, 1);

	Mat depthMap = R*merge;

	depthMap = depthMap.t();
	depthMap = depthMap.reshape(4,depthImg.rows);

	return depthMap;

}
cv::Mat Environment::clearNanInf(cv::Mat matrix)
{
	for (int r = 0; r < matrix.rows; r++)
		for (int c = 0; c < matrix.cols; c++)
		{
			if ( matrix.ptr<double>(r)[c] !=  matrix.ptr<double>(r)[c])
			{
				matrix.ptr<double>(r)[c] = 0;
			}else if (matrix.ptr<double>(r)[c] < 0 || matrix.ptr<double>(r)[c] > 1)
			{
				matrix.ptr<double>(r)[c] = 0;
			}
		}
	return matrix;
}
cv::Mat Environment::map3dTo2d(cv::Mat map)
{
	Mat map2D = Mat(map.size[0],map.size[1], CV_64F, Scalar::all(0));

	for (int r = 0 ; r < map.size[0]; r++)
		for(int c = 0; c < map.size[1]; c++)
			for(int h = 0 ; h < map.size[2]; h++)
			{
				map2D.ptr<double>(r)[c] +=map.ptr<double>(r)[c*map.size[2] + h];
			}

	return map2D;

}
void Environment::calculateNewPan(float &policyPan, float &disPlacement, float prevDisplace)
{
	double panAngle =  keepAngleWithin180(policyPan + disPlacement);
	if ( fabs(panAngle) > _PTConfig.maxPan)
	{
		float newPan =  policyPan + disPlacement;
		float angleDifference = fabs(newPan) - _PTConfig.maxPan;
		if (newPan < 0)
		{
			angleDifference =  -1*angleDifference;
			newPan = -1*_PTConfig.maxPan;
		}
		else
		{
			newPan = _PTConfig.maxPan;
		}

		policyPan = newPan;

		// TODO This line has to be changed with an actual control to turn the robot
		float robotDirNew =keepAngleWithin180(getRobotDir());
		disPlacement = keepAngleWithin180(prevDisplace - robotDirNew);
	}
	else
	{
		policyPan = panAngle;
	}
}

cv::Mat Environment::generateSaliencyMap(){
	Mat aimMap, bpMap,aimMask, imageMasked, salImg, salMap;
	int numBins = 64; // Number of histogram nackprojection
	//String pathToAIMBasis = "../21infomax950.bin";
	String pathToAIMBasis = "../21infomax950.bin";
	float precntileThresh = 95;
	float scaleFactor = 1;
	double aimRate = 0.2;
	double bpRate = 0.8;
	String bpTempPath = "../red.jpg";
	aimMask = Mat(_envImage.rows, _envImage.cols, CV_8U, Scalar::all(1)); // Aim mask for generating the final saliency map
	aimMap = Mat(_envImage.rows, _envImage.cols, CV_8U, Scalar::all(0));
	bpMap = Mat(_envImage.rows, _envImage.cols, CV_8U, Scalar::all(0));
	salMap = Mat(3,_envMapSize, CV_64F, Scalar::all(0));
	Mat envImage = _envImage.clone();

	aimMap = _saliency->getAIM(envImage,precntileThresh, scaleFactor, pathToAIMBasis);

	threshold(aimMap, aimMask, 0,1,THRESH_BINARY);
	for(int r = 0; r < aimMask.rows; r++)
		for(int c = 0; c < aimMask.cols; c++)
		{
			if (aimMask.ptr<uchar>(r)[c]  == 0)
			{
				envImage.ptr<uchar>(r)[c*3] = 0;
				envImage.ptr<uchar>(r)[c*3 +1] = 0;
				envImage.ptr<uchar>(r)[c*3 +2] = 0;
			}
		}
	Mat tempImage = imread(bpTempPath,CV_LOAD_IMAGE_COLOR);

	bpMap = _saliency->getBackProj(envImage,tempImage,"C1C2C3", true, numBins);

	aimMap.convertTo(aimMap, CV_64F);
	bpMap.convertTo(bpMap,CV_64F);
	_saliencyImg = aimMap*aimRate + bpMap*bpRate;
	salImg = (aimMap*aimRate  + bpMap*bpRate)/255;
	Scalar sumSal = sum(salImg);
	salImg /= sumSal[0];
	salImg = clearNanInf(salImg);

	// TODO Transforms the final saliency map to
	//salMap = imageToMap(salImg);
	//float saliencyConf = 0.005;
	//Mat updatedMap = salMap*saliencyConf;
	//return updatedMap;

	return salMap; // Comment out once the transformation is fixed
}

// Updates the probabilities of the search map
void Environment::updateEnvironment()
{
	//TODO : add the saliency and obstacle info
	Point3d robotPos = Point3d(getRobotPos().x/_voxelSize,getRobotPos().y/_voxelSize,_CamConfig.cameraHeight/_voxelSize);
	float robotDir = getRobotDir();
	float pan, tilt;
	getPanTilt(pan,tilt);
	float corrPan = pan + robotDir;
	const double HFOV = (_CamConfig.cameraHorizontalViewAngle/ 2.);
	float recMaxRange =  _recMaxRange/ _voxelSize;
	float recMinRange =  _recMinRange/_voxelSize;
	for (int i = 0; i < _environment3D.size[1]; i++)
	{
		for (int j = 0; j < _environment3D.size[0]; j++)
		{
			for (int k = 0; k < _environment3D.size[2]; k++)
			{
				Point3d p(i, j, k);
				Point3d vec = p - robotPos;
				double length = sqrt(SQR(vec.x) + SQR(vec.y)+SQR(vec.z));
				float angV = acos(vec.z/length)*180/PI;
				float angH = getAngleOfVector(Point2d(robotPos.x,robotPos.y), Point2d(p.x,p.y));


				if ((fabs(corrPan - angH) > HFOV
						&& fabs(corrPan + 360 - angH) > HFOV
						&& fabs(corrPan - 360 - angH) > HFOV)
						||
						fabs( (90 - angV)- tilt) > HFOV)

				{
					break;
				}
				else if (length < recMaxRange && length > recMinRange)
				{
					_environment3D.ptr<double>(j)[i*_environment3D.size[2]+k] = 0;
				}
			}
		}
	}

	//TODO Read the saliency value and update the probability values accordingly
	//Mat saliency3DMap = generateSaliencyMap();

	// Use this function if you want to aggregate saliency values column-wise
	// and use the 2d resulted saliency map
    // Mat saliencyMap2D =  map3dTo2d(saliency3DMap);


	int numNZero = countNonZero(_environment3D);
	double prob = 1./numNZero;
	for (int i = 0; i < _environment3D.size[1]; i++)
	{
		for (int j = 0; j < _environment3D.size[0]; j++)
		{
			for (int k = 0; k < _environment3D.size[2]; k++)
			{
				if(_environment3D.ptr<double>(j)[i*_environment3D.size[2]+k] > 0)
				{
					_environment3D.ptr<double>(j)[i*_environment3D.size[2]+k] = prob;

					//Uncomment once generated the saliency map
					//_environment3D.ptr<double>(j)[i*_environment3D.size[2]+k] = prob +
					// saliency3DMap.ptr<double>(j)[i*_environment3D.size[2]+k];
				}
			}
		}
	}


	//TODO Normalize the environment map if saliency is used
}

// Visualizes the search environment by generating a 2D map, and identifying the location of the robot
Mat Environment::visualizeEnvironment()
{
	Mat map3d = Mat(_environment3D.size[0], _environment3D.size[1], CV_8UC3, Scalar(255,0,0));
	Mat obsMap;
	_obstacleMap.convertTo(obsMap,CV_8UC1);
	// normalize(obsMap,obsMap,50,0,NORM_MINMAX);
	// double minScale, maxScale;
	// int minIdx[3], maxIdx[3];

	//minMaxIdx(_environment3D, &minScale, &maxScale, minIdx, maxIdx);
	//float coef = 200 / maxScale;
	float intensityEnhance = round(_environment3D.size[0]*_environment3D.size[1]*_environment3D.size[2])*100;//*3;
	for (int i = 0; i < _environment3D.size[0]; i++){
		for (int j = 0; j < _environment3D.size[1]; j++){
			for (int k = 0; k < _environment3D.size[2]; k++){
				// uchar obsCol = obsMap.at<uchar>(i, j);
				if (_environment3D.ptr<double>(i)[j*_environment3D.size[2]+k] == 0){
					map3d.ptr<uchar>(i)[j*3] = 0;
					map3d.ptr<uchar>(i)[j*3+1] = 0;//+obsCol;
					map3d.ptr<uchar>(i)[j*3+2] = 0;
					break;
				}
				else if (_environment3D.ptr<double>(i)[j*_environment3D.size[2]+k] > 0)
				{

					uchar col = _environment3D.ptr<double>(i)[j*_environment3D.size[2]+k]*intensityEnhance;
					map3d.ptr<uchar>(i)[j*3] = col;
					map3d.ptr<uchar>(i)[j*3+1] = col;//+obsCol;
					map3d.ptr<uchar>(i)[j*3+2] = col;
					break;
				}
			}
		}
	}
	int robot = _RobConfig.robotRadius/ _voxelSize;
	Point2f robotLoc = getRobotPos();
	rectangle(map3d, Rect((robotLoc.x/_voxelSize) - round(robot/2.), (robotLoc.y/_voxelSize)
			- round(robot/2.), robot, robot), Scalar(0, 0, 255), CV_FILLED);
	return map3d;
}

// Conducts search
void Environment::search()
{
	EnvConfig config;
	// The initial location of the robot in the environment.
	// It can equivalently be read from the robot
	config.robotPos = Point2d(500,500);
	config.robotDir = Point2d(501,500);

	// The resolution of the search space in square mm
	config.voxelSize = 100;

	// The dimensions of the environment
	config.envSize = Point3i(10000,10000,1000);

    // The search method, greedy or lookMove
	config.searchMethod = SearchConfig::lookMove;

	// The look move search threshold to move the robot to the next location to search
	// This values should be set empirically based on:
	// 1- The resolution of the search environment
	// 2- How fast the robot has to move to the next locations
	config.searchThreshold = 0.0003;
	float dirDisplacement = 0;
	Environment e(config);

	// build a list of all possible pan and tilt angle combinations
	vector<CameraViewDirection> views = e.buildListOfViewDirections();

	for(int i = 0 ; i < 30;i++)
	{

		vector<BestPolicy> policy = e.chooseBestAction(views);

		if (policy[0].distance > 0)
		{
			float robotDirPrev = Environment::keepAngleWithin180(e.getRobotDir());

			// moves the robot to the next location.
			// This has to be replaced with navigation
			e.setRobotPos(Point2d(policy[0].p.x,policy[0].p.y));

			// This part is used when the search method is greedy.
			// So the calculation of the next best policy at the time is based on the current
			// direction of the robot. Since the robot may move in the environment, its next direction
			// might not be the same. So the following lines calculate the new pan angle and if
			// the new angle is beyond the pan torque limits, it turns the robot to compensate for it

			float robotDirNew = Environment::keepAngleWithin180(e.getRobotDir());
			dirDisplacement = Environment::keepAngleWithin180(robotDirPrev - robotDirNew);
			e.calculateNewPan(policy[i].direction.Pan, dirDisplacement, robotDirPrev);


			if (e.getMethod() == SearchConfig::lookMove)
			{
				continue;
			}
		}

		// TODO in practice has to be replaced with actual controls for the robot
		e.setPan(policy[0].direction.Pan);
		e.setTilt(policy[0].direction.Tilt);

		// TODO replace the following lines by reading the image from the camera

		// uncomment this for a sample saliency result

		String gg = "../testimg.png";
		_envImage = imread(gg,CV_LOAD_IMAGE_COLOR);
		generateSaliencyMap();


		// TODO Perform recognition to look for the object

		e.updateEnvironment();
		Mat env2D = e.visualizeEnvironment();

		//Display the 2d representation of the 3D environment map
		resize(env2D,env2D,Size(640,480),2,2,INTER_CUBIC);
		imshow("env2D", env2D);
		waitKey(1000);

	}
}
int main()
{
	Environment e;
	e.search();
}
