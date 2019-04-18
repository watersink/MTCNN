#include "detect_face.h"

void  bbreg(vector<RESULT>& result)
{
#pragma omp parallel for
	for (unsigned int i = 0; i < result.size(); i++)
	{
		float xend = result[i].boudingboxes.x + result[i].boudingboxes.width;
		float yend = result[i].boudingboxes.y + result[i].boudingboxes.height;
		result[i].boudingboxes.x = result[i].boudingboxes.x + result[i].reg[1] * result[i].boudingboxes.width;
		result[i].boudingboxes.y = result[i].boudingboxes.y + result[i].reg[0] * result[i].boudingboxes.height;
		result[i].boudingboxes.width = xend + result[i].reg[3] * result[i].boudingboxes.width - result[i].boudingboxes.x;
		result[i].boudingboxes.height = yend + result[i].reg[2] * result[i].boudingboxes.height - result[i].boudingboxes.y;

	}

}
