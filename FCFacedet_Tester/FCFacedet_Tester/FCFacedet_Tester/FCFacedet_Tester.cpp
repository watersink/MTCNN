// FCFacerec_Tester.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

#include "FCFacedet.h"


int main(int argc, char* argv[])
{
	argc = 3;
	argv[1] = "0";
	argv[2] = "peter.jpg";


	if (argc < 3)
	{
		cout << "usage: *.exe 0[1] img[video]_path" << endl;
		waitKey(0);
		return -1;
	}

	bool bUseGPU = false;
	if (!strcmp("1", argv[1]))
		bUseGPU = true;

	CFCFacedet* FCFacedet = new CFCFacedet;

	bool bInitRet = FCFacedet->FCFInit("./model", bUseGPU);
	if (!bInitRet)
	{
		cout << "FCFInit Failed!" << endl;
		return -1;
	}

	VideoCapture capture;

//	if (!strcmp("0", argv[2]))
//		capture.open(0);
//	else
//		capture.open(argv[2]);

//	if (!capture.isOpened())
//		cout << "fail to open!" << endl;


	cv::Mat matImg1;

	while (1)
	{
		//if (!capture.read(matImg1))
		//{
		//	cout << "read frame failed" << endl;
		//	break;
		//}
		matImg1 = cv::imread(argv[2]);

		clock_t start1, end1;
		start1 = clock();

		ST_IMAGE_DATA stImg1(matImg1.cols, matImg1.rows, matImg1.channels());
		stImg1.data = matImg1.data;

		std::vector<ST_FACE_DATA> vctFaces;
		FCFacedet->FCFGetFaces(stImg1, vctFaces, 40);
		end1 = (double)(1000 * (clock() - start1) / CLOCKS_PER_SEC);
		cout << "      time1: " << end1 << "ms" << ", face count: " << vctFaces.size() << endl;

		for (unsigned int u = 0; u < vctFaces.size(); u++)
		{
			rectangle(matImg1, cv::Rect(vctFaces[u].x_pos, vctFaces[u].y_pos, vctFaces[u].width, vctFaces[u].height), Scalar(255, 0, 0), 2);
			for (int k = 0; k < 5; k++)
			{
				circle(matImg1, cv::Point(vctFaces[u].landmark[k * 2], vctFaces[u].landmark[k * 2 + 1]), 2, Scalar(255, 255, 0), 4);
			}
		}

		imshow("Results", matImg1);
		waitKey(10);

	}

	waitKey(0);

	delete FCFacedet;

	return 0;
}

