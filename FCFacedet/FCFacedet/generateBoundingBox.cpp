#include "detect_face.h"

vector<RESULT> generateBoundingBox(Mat map, vector<Mat> reg, float scale, float t)
{
	//map为CV32FC1，有1 个，reg为CV32FC1，有4个
	//use heatmap to generate bounding boxes
	vector<RESULT> result;
	
	int stride = 2;
	int cellsize = 12;

	Mat dx1 = reg[0];
	Mat dy1 = reg[1];
	Mat dx2 = reg[2];
	Mat dy2 = reg[3];
	
	vector<Point2f> xy;
	for (int i = 0; i < map.rows;i++)
	{
		for (int j = 0; j < map.cols; j++)
		{	
			if (map.at<float>(i, j) >= t)
			{
				xy.push_back(Point2f(j, i));
			}
		}
	}
	for (unsigned int i = 0; i < xy.size(); i++)
	{
		RESULT resulttmp;
		resulttmp.reg[0] = dx1.at<float>(Point2f(xy[i]));
		resulttmp.reg[1] = dy1.at<float>(Point2f(xy[i]));
		resulttmp.reg[2] = dx2.at<float>(Point2f(xy[i]));
		resulttmp.reg[3] = dy2.at<float>(Point2f(xy[i]));

		resulttmp.score = map.at<float>(Point2f(xy[i]));

		resulttmp.boudingboxes.x = floor((stride*(xy[i].x) + 1) / scale) - 1;
		resulttmp.boudingboxes.y = floor((stride*(xy[i].y) + 1) / scale) - 1;
		resulttmp.boudingboxes.width = floor((stride*(xy[i].x) + cellsize ) / scale) - 1 - resulttmp.boudingboxes.x;
		resulttmp.boudingboxes.height = floor((stride*(xy[i].y) + cellsize ) / scale) - 1 - resulttmp.boudingboxes.y;

		result.push_back(resulttmp);
		
	}
	
	return result;

}
