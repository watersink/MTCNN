#include "detect_face.h"
#include <algorithm>

typedef pair<int, float> PAIR;
int cmp(const PAIR& x, const PAIR& y)
{
	return x.second < y.second;
}
vector<int> nms(vector<RESULT> boxes, float threshold, string type)
{
	vector<int>pick;
	if (boxes.size() == 0)
		return pick;

	vector<int> x1;
	vector<int> y1;
	vector<int> x2;
	vector<int> y2;
	vector<PAIR> s;
	vector<int> area;
	vector<int> is_suppressed;
	for (unsigned int i = 0; i < boxes.size(); i++)
	{
		x1.push_back(boxes[i].boudingboxes.x);
		y1.push_back(boxes[i].boudingboxes.y);
		x2.push_back(boxes[i].boudingboxes.x + boxes[i].boudingboxes.width);
		y2.push_back(boxes[i].boudingboxes.y + boxes[i].boudingboxes.height);

		s.push_back(make_pair(i, boxes[i].score));

		area.push_back((boxes[i].boudingboxes.width + 1) * (boxes[i].boudingboxes.height + 1));

		is_suppressed.push_back(0);
	}

	sort(s.begin(), s.end(), cmp);

	vector<PAIR> s_copy;
	while (s.size() > 0)
	{
		int last = s.size();
		int i = s[last - 1].first;
		s[last - 1].first = -1;
		pick.push_back(i);
		last = last - 1;
		vector<float> o(last), xx1(last), yy1(last), xx2(last), yy2(last), w(last), h(last), inter(last);
		for (int m = last - 1; m >= 0; m--)
		{
			int idx = s[m].first;
			xx1[m] = max(x1[i], x1[idx]);
			yy1[m] = max(y1[i], y1[idx]);
			xx2[m] = min(x2[i], x2[idx]);
			yy2[m] = min(y2[i], y2[idx]);
			w[m] = max(0.0, xx2[m] - xx1[m] + 1.0);
			h[m] = max(0.0, yy2[m] - yy1[m] + 1.0);
			inter[m] = w[m] * h[m];

			if (type == "Min")
			{
				o[m] = inter[m] / min(area[i], area[idx]);
			}
			else
			{
				o[m] = inter[m] / (area[i] + area[idx] - inter[m]);
			}

			if (o[m] > threshold)
			{
				//printf("s[%d] = %d, del %d\n", m, s[m].first, m);
				s[m].first = -1;
			}
		}

		s_copy.clear();
		vector<PAIR>::iterator itr = s.begin();
		for (; itr != s.end(); itr++)
		{
			if (itr->first != -1)
			{
				s_copy.push_back(*itr);
			}
		}
		s = s_copy;
	}

	return pick;
}



/*
vector<int> nms(vector<RESULT> boxes, float threshold, string type)
{
	vector<int>pick;
	if (boxes.size() == 0)
		return pick;

	vector<int> x1;
	vector<int> y1;
	vector<int> x2;
	vector<int> y2;
	vector<PAIR> s;
	vector<int> area;
	vector<int> is_suppressed;
	for (int i = 0; i < boxes.size();i++)
	{
		x1.push_back(boxes[i].boudingboxes.x);
		y1.push_back(boxes[i].boudingboxes.y);
		x2.push_back(boxes[i].boudingboxes.x + boxes[i].boudingboxes.width);
		y2.push_back(boxes[i].boudingboxes.y + boxes[i].boudingboxes.height);

		s.push_back(make_pair(i,boxes[i].score));

		area.push_back((boxes[i].boudingboxes.width+1) * (boxes[i].boudingboxes.height+1));

		is_suppressed.push_back(0);
	}

	sort(s.begin(), s.end(), cmp);

	for (int i = 0; i < boxes.size(); i++)                // 循环所有窗口   
	{
		if (!is_suppressed[s[i].first])           // 判断窗口是否被抑制   
		{
			pick.push_back(s[i].first);
			for (int j = i + 1; j < boxes.size(); j++)    // 循环当前窗口之后的窗口   
			{
				if (!is_suppressed[s[j].first])   // 判断窗口是否被抑制   
				{
					int xx1 = max(x1[s[i].first], x1[s[j].first]);                     // 求两个窗口左上角x坐标最大值   
					int xx2 = min(x2[s[i].first], x2[s[j].first]);     // 求两个窗口右下角x坐标最小值   
					int yy1 = max(y1[s[i].first], y1[s[j].first]);                     // 求两个窗口左上角y坐标最大值   
					int yy2 = min(y2[s[i].first], y2[s[j].first]);     // 求两个窗口右下角y坐标最小值   
					int w = std::max(0, xx2 - xx1 + 1);     // 计算两矩形重叠的宽度   
					int h = std::max(0, yy2 - yy1 + 1);     // 计算两矩形重叠的高度  
					int inter = w*h;
					//if (w > 0 && h > 0)
					{
						float o;
						if (type == "Min")
							o = (float)inter / std::min((float)area[i], (float)area[j]); //交 / 所有最小   // 计算重叠的比率   
						else
							o = (float)inter / ((float)area[i] + (float)area[j] - (float)inter);
						if (o > threshold )   // 判断重叠比率是否超过重叠阈值   
						{
							is_suppressed[s[j].first] = 1;     // 将窗口j标记为抑制   
						}
					}
				}
			}
		}
	}

	return pick;


}*/
