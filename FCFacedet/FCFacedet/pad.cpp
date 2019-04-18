#include "detect_face.h"

vector<PAD_RESULT> pad(vector<RESULT> result, int w, int h)
{
	//compute the padding coordinates(pad the bounding boxes to square)
	vector<PAD_RESULT> pad_result(result.size());
#pragma omp parallel for
	for (unsigned int i = 0; i < result.size();i++)
	{
		

		pad_result[i].x = floor(result[i].boudingboxes.x);
		pad_result[i].y = floor(result[i].boudingboxes.y);
		pad_result[i].ex = floor(result[i].boudingboxes.x + result[i].boudingboxes.width);
		pad_result[i].ey = floor(result[i].boudingboxes.y + result[i].boudingboxes.height);

		pad_result[i].tmpw = pad_result[i].ex - pad_result[i].x+1;//floor(result[i].boudingboxes.width);
		pad_result[i].tmph = pad_result[i].ey - pad_result[i].y+1;//floor(result[i].boudingboxes.height);

		pad_result[i].edx = pad_result[i].tmpw;
		pad_result[i].edy = pad_result[i].tmph;

		if (pad_result[i].ex > w)
		{
			pad_result[i].edx = pad_result[i].ex*(-1) + w + pad_result[i].tmpw;
			pad_result[i].ex = w;
		}
		if (pad_result[i].ey> h)
		{
			pad_result[i].edy = pad_result[i].ey*(-1) + h + pad_result[i].tmph;
			pad_result[i].ey = h;
		}
		if (pad_result[i].x<0)
		{
			pad_result[i].dx = 1 - pad_result[i].x;
			pad_result[i].x=0;
		}else
			pad_result[i].dx = 0;

		if (pad_result[i].y < 0)
		{
			pad_result[i].dy = 1 - pad_result[i].y;
			pad_result[i].y = 0;
		}else
			pad_result[i].dy = 0;

	}
	return pad_result;

}
