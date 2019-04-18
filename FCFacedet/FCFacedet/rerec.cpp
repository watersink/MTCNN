#include "detect_face.h"


void  rerec(vector<RESULT>& result)
{
	vector <float>l(result.size());
#pragma omp parallel for
	for (unsigned int i = 0; i < result.size(); i++)
	{
		l[i] = std::max(result[i].boudingboxes.width, result[i].boudingboxes.height);
		result[i].boudingboxes.x = result[i].boudingboxes.x + result[i].boudingboxes.width*0.5 - l[i] * 0.5;
		result[i].boudingboxes.y = result[i].boudingboxes.y + result[i].boudingboxes.height*0.5 - l[i] * 0.5;
		result[i].boudingboxes.width = l[i];
		result[i].boudingboxes.height = l[i];
	}
}
