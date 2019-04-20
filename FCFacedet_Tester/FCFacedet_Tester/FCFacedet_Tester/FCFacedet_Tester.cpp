// FCFacerec_Tester.cpp : Defines the entry point for the console application.
//

#include "FCFacedet.h"



#define USE_OPENCV 1
#define CPU_ONLY 1


#include <iostream>
#include<string>



#include <caffe/caffe.hpp>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include "head.h"
using namespace caffe;
using namespace cv;



int main(int argc, char* argv[])
{

	CFCFacedet* FCFacedet = new CFCFacedet;

	bool bInitRet = FCFacedet->FCFInit("./model", 0);
	if (!bInitRet)
	{
		cout << "FCFInit Failed!" << endl;
		return -1;
	}


	char imagename[1024]="test.jpg";
		
		
	cv::Mat matImg;
	matImg = cv::imread(imagename);

	ST_IMAGE_DATA stImg(matImg.cols, matImg.rows, matImg.channels());
	stImg.data = matImg.data;

	vector<ST_FACE_DATA> vctFaces;
	FCFacedet->FCFGetFaces(stImg, vctFaces, 10);
		

	float facial5points[10];
	
	for (int i = 0; i < vctFaces.size(); i++)
	{
		rectangle(matImg, cv::Rect(vctFaces[i].x_pos, vctFaces[i].y_pos, vctFaces[i].width, vctFaces[i].height), Scalar(255, 0, 0), 2);
		for (int k = 0; k < 5; k++)
		{
			facial5points[k] = vctFaces[i].landmark[k * 2];
			facial5points[5 + k] = vctFaces[i].landmark[k * 2 + 1];
			circle(matImg, cv::Point(vctFaces[i].landmark[k * 2], vctFaces[i].landmark[k * 2 + 1]), 2, Scalar(255, 255, 0), 4);
		}
	}

	imshow("result", matImg);
	waitKey();

	

	delete FCFacedet;

	return 0;
}

