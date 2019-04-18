#define USE_OPENCV 1
#define CPU_ONLY 1
//#define USE_CUDNN 1

#include <caffe/caffe.hpp>
#include <vector>
#include "omp.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace caffe;
using namespace cv;





struct  RESULT
{
	Rect_<float> boudingboxes;
	Point2f points[5];
	float score;

	float reg[4];
};
struct  PAD_RESULT
{
	int dy;
	int edy;
	int dx;
	int edx;
	int y;
	int ey;
	int x;
	int ex;
	int tmpw;
	int tmph;
};

vector<RESULT>  detect_face(Mat& img,
	int minsize,
	caffe::shared_ptr<Net<float> >& PNet,
	caffe::shared_ptr<Net<float> >& RNet,
	caffe::shared_ptr<Net<float> >& ONet,
	float threshold[3],
	bool fastresize,
	float factor);

void  bbreg(vector<RESULT>& result);
void  rerec(vector<RESULT>& result);
vector<PAD_RESULT> pad(vector<RESULT> result, int w, int h);
vector<RESULT> generateBoundingBox(Mat map, vector<Mat> reg, float scale, float t);
vector<int> nms(vector<RESULT> boxes, float threshold, string type);

void WrapInputLayerRNetONet(caffe::shared_ptr<Net<float> >& net_, std::vector<cv::Mat>* input_channels);
