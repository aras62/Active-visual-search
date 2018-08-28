#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/Image.h>
#include "std_msgs/Float32.h"
#include <std_msgs/Float64.h>
#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/Float64MultiArray.h"
#include <sensor_msgs/fill_image.h>
#include <boost/algorithm/string.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include "ros/callback_queue.h"
#include <saliency/GetAIM.h>
#include <saliency/GetBackProj.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

enum colorSpace {RGB=0, HSV, Lab, Luv, HSI, HSL, CMY, C1C2C3, COPP, YCrCb, YIQ, XYZ, UVW, YUV,
    OPP, NOPP, xyY, rg, YES, I1I2I3};
#define PI 3.14159265359
#define SQR(X) ((X)*(X))
class Saliency
{
public:
	Saliency();
	virtual ~Saliency();
	void resetAIM();
	//****************************** Utilities ******************************
	cv::Mat imageConversion(cv::Mat inputImg,colorSpace type, bool norm = true);
	cv::Mat normalizeImage(cv::Mat RGBImage);
	void normalizeHistogram(cv::Mat &histogram);
	cv::Mat percentileThreshold(cv::Mat salMap, double percentile);

	//****************************** Methods ******************************
	cv::Mat generateBackProjection(cv::Mat image, cv::Mat temp);

	void loadBasis(std::string filename = "../21infomax950.bin");
	cv::Mat runAIM();
	cv::Mat generateAIMMap(cv::Mat imageName, float scale_factor, std::string basisName = "../21infomax950.bin");

	//Call Back functions
	bool GetAIMMap(saliency::GetAIM::Request& req, saliency::GetAIM::Response& res);
	bool GetBackProjMap(saliency::GetBackProj::Request& req, saliency::GetBackProj::Response& res);
	void QueueThread();
	void ROSNodeInit();
	void InitRosTopics();
	sensor_msgs::Image fillImageMsgs(cv::Mat image, std::string imgName);
	cv::Mat getImageFromMsg(sensor_msgs::Image msg);
public:
	static Saliency*_instance;
	std::vector<std::string> _colors;
	cv::Mat adj_sm;
private:

	//******************* AIM params *********************
	cv::Mat **kernels;
	float* data;
	float scale;
	std::vector<cv::Mat> channels;
	cv::Mat image, temp, sm, hist;
	cv::Mat *aim_temp;
	int num_kernels, kernel_size, num_channels;
	bool gotKernel;
	double maxVal, minVal, max_aim, min_aim;
	int counter;

	//******************* BP Params ***********************
	int num_bins;
	std::unique_ptr<ros::NodeHandle> rosNode;
	ros::ServiceServer getAIMSrv, getBackProjSrv;
	std::string  getAIMService, getBackProjService;
	ros::CallbackQueue rosQueue;
	boost::thread callbackQueueThread;
	std::string namespace_;

};
