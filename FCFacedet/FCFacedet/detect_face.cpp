#include "detect_face.h"
#include <fstream>

void WrapInputLayer(caffe::shared_ptr<Net<float> >& net_,std::vector<cv::Mat>* input_channels)
{
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height(); 
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) 
	{
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Preprocess(caffe::shared_ptr<Net<float> >& net_, 
	int num_channels_, 
	cv::Size input_geometry_,
	const cv::Mat& img, 
	std::vector<cv::Mat>* input_channels)
{
	/* Convert the input image to the input image format of the network. */

	cv::Mat sample;
	cv::cvtColor(img, sample, CV_BGR2RGB);
	
	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample.convertTo(sample_float, CV_32FC3);
	else
		sample.convertTo(sample_float, CV_32FC1);


	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample_float, sample_resized, input_geometry_, 0.0, 0.0, INTER_AREA);
	else
		sample_resized = sample_float;


	subtract(sample_resized, 127.5, sample_resized);
	sample_resized = sample_resized.mul(0.0078125);

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	//这个就是送进去的图
	cv::split(sample_resized, *input_channels);

	//输出采样的图片sample_resized
	//std::ofstream oof("1.txt", ios::trunc);
	//oof << sample_resized.rows << "  " << sample_resized.cols << std::endl;
	//for (int k = 0; k < 3; k++)
	//{
	//	oof << "---------------------------------------" << k << std::endl;
	//	for (int i = 0; i < 3;i++)// sample_resized.rows; i++)
	//	for (int j = 0; j < 3;j++)// sample_resized.cols; j++)
	//	{
	//		oof << sample_resized.at<Vec3f>(i, j)[k] << std::endl;
	//	}
	//}
	//oof.close();


	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}


void WrapInputLayerRNetONet(caffe::shared_ptr<Net<float> >& net_, std::vector<cv::Mat>* input_channels)
{
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	int num = input_layer->num();
	float* input_data = input_layer->mutable_cpu_data();
	for (int k = 0; k < num; k++)
	{
		for (int i = 0; i < input_layer->channels(); ++i)
		{
			cv::Mat channel(height, width, CV_32FC1, input_data);
			input_channels->push_back(channel);
			input_data += width * height;
		}

	}
	
}
void PreprocessRNetONet(caffe::shared_ptr<Net<float> >& net_,
	vector<Mat>& temping,
	std::vector<cv::Mat>* input_channels)
{
	Blob<float>* input_layer = net_->input_blobs()[0];		

	for (unsigned int i = 0; i < temping.size();i++)
	{
		Mat tmp = temping[i];
		cv::cvtColor(tmp, tmp, CV_BGR2RGB);

		subtract(tmp, 127.5, tmp);
		tmp = tmp.mul(0.0078125);

		vector<Mat>temp_inputChannels;
		cv::split(tmp, temp_inputChannels);
		temp_inputChannels[0].copyTo(input_channels->at(i * input_layer->channels() + 0));
		temp_inputChannels[1].copyTo(input_channels->at(i * input_layer->channels() + 1));
		temp_inputChannels[2].copyTo(input_channels->at(i * input_layer->channels() + 2));
		
	}

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}



vector<RESULT>  detect_face(Mat& img,
	int minsize,
	caffe::shared_ptr<Net<float> >& PNet,
	caffe::shared_ptr<Net<float> >& RNet,
	caffe::shared_ptr<Net<float> >& ONet,
	float threshold[3],
	bool fastresize,
	float factor)
{
	img = img.t();
	
	vector<RESULT> total_result;
	vector<RESULT> total_resultmid;
	vector<RESULT> total_resultend;

	int factor_count = 0;

	int h = img.rows;
	int w = img.cols;

	float minl = std::min(h,w);

	
	float m = 12.0 / minsize;
	minl = minl*m;
	
	//creat scale pyramid
	vector<float>scales;
	while (minl >= 12)
	{
		scales.push_back(m*powf(factor ,factor_count));
		minl = minl*factor;
		factor_count = factor_count + 1;
	}
	
	//first stage
	vector<RESULT> total_result1;
	for (unsigned int j = 0; j < scales.size(); j++)
	{
		float scale = scales[j];
		int hs = ceil(h*scale);
		int ws = ceil(w*scale);
		
		int PNet_num_channels_ = 3;
		cv::Size PNet_input_geometry_ = Size(ws, hs);


		Blob<float>* PNet_input_layer = PNet->input_blobs()[0];
		/* Forward dimension change to all layers. */
		
		PNet_input_layer->Reshape(1, 3, hs, ws);
		//PNet->Reshape();

		std::vector<cv::Mat> PNet_input_channels;
		WrapInputLayer(PNet, &PNet_input_channels);
		Preprocess(PNet, PNet_num_channels_, PNet_input_geometry_, img, &PNet_input_channels);		
		
		PNet->Forward();


		////输出第一个卷积层的结果---------------------------------------------------------------------
		//shared_ptr<Blob<float> >  conv1Blob = PNet->blob_by_name("conv1");   
		//std::cout << "测试图片的特征响应图的形状信息为：" << conv1Blob->shape_string() << std::endl;   	
		//std::ofstream ofs("conv1.txt", ios::trunc);
		//ofs<< "测试图片的特征响应图的形状信息为：" << conv1Blob->shape_string() << std::endl;

		//int width = conv1Blob->shape(3);  //响应图的高度	
		//int height = conv1Blob->shape(2);  //响应图的宽度	
		//int channel = conv1Blob->shape(1);  //通道数
		//int num = conv1Blob->shape(0);      //个数	
		//
		//for (int kkk = 0; kkk < channel;kkk++)
		//{
		//		for (int iii = 0; iii<height; iii++)
		//		{
		//			for (int jjj = 0; jjj<width; jjj++)
		//			{
		//				float value = conv1Blob->data_at(0, kkk, iii, jjj);
		//				ofs << value <<std::endl;		
		//			}
		//		}		
		//}
		//
		//ofs.close();
		////输出第一个卷积层的结果---------------------------------------------------------------------








		/* Copy the output layer to a std::vector */
		Blob<float>* output_layer0 = PNet->output_blobs()[0];
		Blob<float>* output_layer1 = PNet->output_blobs()[1];
		
		vector<Mat> reg;
		for (int kkk = 0; kkk < 4; kkk++)
		{
		Mat regtmp(output_layer0->height(), output_layer0->width(), CV_32FC1);
		memcpy(regtmp.data, output_layer0->cpu_data() + output_layer0->width()*output_layer0->height()*kkk, output_layer0->width()*output_layer0->height()*sizeof(float));
		reg.push_back(regtmp);
		}

		cv::Mat map(output_layer1->height(), output_layer1->width(), CV_32FC1);
		memcpy(map.data, output_layer1->cpu_data() + output_layer1->width()*output_layer1->height(), output_layer1->width()*output_layer1->height()*sizeof(float));

		vector<RESULT>result1;
		result1=generateBoundingBox(map,reg, scale, threshold[0]);
	
		//inter - scale nms
		vector<int> pick;
		pick = nms(result1, 0.5, "Union");

		vector<RESULT>result;
		for (unsigned int pick_iter = 0; pick_iter < pick.size(); pick_iter++)
		{
			result.push_back(result1[pick[pick_iter]]);
		}
		if (result.size() != 0)
		{
			for (unsigned int i = 0; i < result.size(); i++)
			{
				total_result1.push_back(result[i]);
			}
		}
		
	}

	
	vector<PAD_RESULT> pad_result;
	if (total_result1.size() != 0)
	{
		vector<int> picktmp;
		picktmp = nms(total_result1, 0.7, "Union");
		for (unsigned int pick_iter = 0; pick_iter < picktmp.size(); pick_iter++)
		{
			total_result.push_back(total_result1[picktmp[pick_iter]]);
		}
		for (unsigned int i = 0; i < total_result.size(); i++)
		{
			int regw = total_result[i].boudingboxes.width;
			int regh = total_result[i].boudingboxes.height;
			int x2 = total_result[i].boudingboxes.x + total_result[i].boudingboxes.width;
			int y2 = total_result[i].boudingboxes.y + total_result[i].boudingboxes.height;

			total_result[i].boudingboxes.x = total_result[i].boudingboxes.x + total_result[i].reg[1] * regw;
			total_result[i].boudingboxes.y = total_result[i].boudingboxes.y + total_result[i].reg[0] * regh;
			
			total_result[i].boudingboxes.width = x2 + total_result[i].reg[3] * regw - total_result[i].boudingboxes.x;
			total_result[i].boudingboxes.height = y2 + total_result[i].reg[2] * regh - total_result[i].boudingboxes.y;

		}
		rerec(total_result);
		pad_result = pad(total_result, w, h);
	}
	int numbox = (int)total_result.size();
	
	
// second stage
	if (numbox > 0)
	{
		vector<Mat>temping;
		for (int k = 0; k < numbox;k++)
		{
			Mat tmp(pad_result[k].tmph, pad_result[k].tmpw, CV_32FC3);
			
			img(Range(pad_result[k].y, pad_result[k].ey), Range(pad_result[k].x, pad_result[k].ex)).copyTo(tmp);
			tmp.convertTo(tmp, CV_32FC3);
			resize(tmp, tmp, Size(24, 24), 0.0, 0.0, INTER_CUBIC);

// 			subtract(tmp, 127.5, tmp);
// 			tmp = tmp.mul(0.0078125);

			temping.push_back(tmp);
		}
		
		Blob<float>* RNet_input_layer = RNet->input_blobs()[0];
		// Forward dimension change to all layers. 

		RNet_input_layer->Reshape(numbox, 3, 24, 24);
		//RNet->Reshape();

		std::vector<cv::Mat> RNet_input_channels;
		WrapInputLayerRNetONet(RNet, &RNet_input_channels);
		PreprocessRNetONet(RNet, temping, &RNet_input_channels);
		RNet->Forward();






		Blob<float>* output_layer0 = RNet->output_blobs()[0];
		Blob<float>* output_layer1 = RNet->output_blobs()[1];
		
		Mat regtmp(output_layer0->num(), 4, CV_32FC1);
		memcpy(regtmp.data, output_layer0->cpu_data(), 4 * output_layer0->num()*sizeof(float));
		Mat reg = regtmp.t();
		
		vector <float>score;
		for (int i = 0; i < output_layer1->count(); i++)
		{
			if (i % 2 != 0)
				score.push_back(*(output_layer1->cpu_data() + i));
		}
		vector<int>pass;
		for (unsigned int i = 0; i < score.size();i++)
		{
			if (score[i]>threshold[1])
			{
				pass.push_back(i);
			}
		}
		vector<RESULT> total_resulttmp(pass.size());
		for (unsigned int i = 0; i < pass.size();i++)
		{
			total_resulttmp[i].boudingboxes=total_result[pass[i]].boudingboxes;
			total_resulttmp[i].score = score[pass[i]];

			total_resulttmp[i].reg[0] = reg.at<float>(0, pass[i]);
			total_resulttmp[i].reg[1] = reg.at<float>(1, pass[i]);
			total_resulttmp[i].reg[2] = reg.at<float>(2, pass[i]);
			total_resulttmp[i].reg[3] = reg.at<float>(3, pass[i]);

#ifdef _DRAW_DET
			rectangle(img, total_resulttmp[i].boudingboxes, Scalar(0, 0, 255), 2);
#endif
		}

		
		
		if (total_resulttmp.size()>0)
		{
			vector<int> picktmp;
			picktmp = nms(total_resulttmp, 0.7, "Union");
			for (unsigned int i = 0; i < picktmp.size();i++)
			{
				total_resultmid.push_back(total_resulttmp[picktmp[i]]);
			}
			bbreg(total_resultmid);
			rerec(total_resultmid);
		}
		unsigned int numbox = total_resultmid.size();

	
	
		
		//third stage
		if (numbox>0)
		{
			pad_result = pad(total_resultmid, w, h);

			vector<Mat>temping;
			for (unsigned int k = 0; k < numbox; k++)
			{
				Mat tmp(pad_result[k].tmph, pad_result[k].tmpw, CV_32FC3);

				img(Range(pad_result[k].y, pad_result[k].ey), Range(pad_result[k].x, pad_result[k].ex)).copyTo(tmp);
				tmp.convertTo(tmp, CV_32FC3);
				resize(tmp, tmp, Size(48, 48), 0.0, 0.0, INTER_CUBIC);

				temping.push_back(tmp);
			}

			Blob<float>* ONet_input_layer = ONet->input_blobs()[0];
			// Forward dimension change to all layers. 

			ONet_input_layer->Reshape(numbox, 3, 48, 48);
			//ONet->Reshape();

			std::vector<cv::Mat> ONet_input_channels;
			WrapInputLayerRNetONet(ONet, &ONet_input_channels);
			PreprocessRNetONet(ONet, temping, &ONet_input_channels);
			ONet->Forward();

			Blob<float>* output_layer0 = ONet->output_blobs()[0];
			Blob<float>* output_layer1 = ONet->output_blobs()[1];
			Blob<float>* output_layer2 = ONet->output_blobs()[2];


			Mat regtmp(output_layer0->num(), 4, CV_32FC1);
			memcpy(regtmp.data, output_layer0->cpu_data(), 4 * output_layer0->num()*sizeof(float));
			Mat reg = regtmp.t();

			Mat pointstmp(output_layer1->num(), 10, CV_32FC1);
			memcpy(pointstmp.data, output_layer1->cpu_data(), 10 * output_layer1->num()*sizeof(float));
			Mat points = pointstmp.t();

			vector <float>score;
			for (int i = 0; i < output_layer2->count(); i++)
			{
				if (i % 2 != 0)
					score.push_back(*(output_layer2->cpu_data() + i));
			}
			
			vector<int>pass;
			for (unsigned int i = 0; i<score.size();i++)
			{
				if (score[i]>threshold[2])
				{
					pass.push_back(i);
				}
			}
			
			vector<RESULT> total_resulttmp2(pass.size());
			for (unsigned int i = 0; i < pass.size(); i++)
			{
				total_resulttmp2[i].boudingboxes=total_resultmid[pass[i]].boudingboxes;
				total_resulttmp2[i].score = score[pass[i]];
				total_resulttmp2[i].points[0] = Point2f(points.at<float>(0, pass[i]), points.at<float>(5, pass[i]));
				total_resulttmp2[i].points[1] = Point2f(points.at<float>(1, pass[i]), points.at<float>(6, pass[i]));
				total_resulttmp2[i].points[2] = Point2f(points.at<float>(2, pass[i]), points.at<float>(7, pass[i]));
				total_resulttmp2[i].points[3] = Point2f(points.at<float>(3, pass[i]), points.at<float>(8, pass[i]));
				total_resulttmp2[i].points[4] = Point2f(points.at<float>(4, pass[i]), points.at<float>(9, pass[i]));

				total_resulttmp2[i].reg[0] = reg.at<float>(0, pass[i]);
				total_resulttmp2[i].reg[1] = reg.at<float>(1, pass[i]);
				total_resulttmp2[i].reg[2] = reg.at<float>(2, pass[i]);
				total_resulttmp2[i].reg[3] = reg.at<float>(3, pass[i]);
			}
			for (unsigned int i = 0; i<total_resulttmp2.size();i++)
			{
				int w = total_resulttmp2[i].boudingboxes.width;
				int h = total_resulttmp2[i].boudingboxes.height;
				for (int j = 0; j < 5; j++)
				{
					int x = w*total_resulttmp2[i].points[j].y;
					int y = h*total_resulttmp2[i].points[j].x;
					total_resulttmp2[i].points[j].x = x + total_resulttmp2[i].boudingboxes.x - 1;
					total_resulttmp2[i].points[j].y = y + total_resulttmp2[i].boudingboxes.y - 1;
				}			
			}

			if (total_resulttmp2.size()>0)
			{
				bbreg(total_resulttmp2);
				vector<int>pick;
				pick = nms(total_resulttmp2, 0.7, "Min");
				for (unsigned int i = 0; i < pick.size();i++)
				{
					total_resultend.push_back(total_resulttmp2[pick[i]]);
				}
			}
		}
	}
	
return total_resultend;
}
