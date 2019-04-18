// FCFacerec.cpp : Defines the exported functions for the DLL application.
//
#include "FCFacedet.h"

#include "detect_face.h"
#include "head.h"

//#define DEBUG_PRINT_TIME

using namespace std;

static bool bIsGoogleLoggingInitialized = false;


bool InitModels(std::string caffe_model_path,
	caffe::shared_ptr<caffe::Net<float>>& PNet,
	caffe::shared_ptr<caffe::Net<float>>& RNet,
	caffe::shared_ptr<caffe::Net<float>>& ONet);

bool GetFaces(cv::Mat& img, std::vector<ST_FACE_DATA>& vctFaces, int minFaceSize,
	caffe::shared_ptr<caffe::Net<float>>& PNet,
	caffe::shared_ptr<caffe::Net<float>>& RNet,
	caffe::shared_ptr<caffe::Net<float>>& ONet);

// This is the constructor of a class that has been exported.
// see FCFacerec.h for the class definition
CFCFacedet::CFCFacedet()
{
	return;
}

CFCFacedet::~CFCFacedet()
{
	if (FCPBX)
	{
		delete (caffe::shared_ptr<caffe::Net<float>>*)FCPBX;
		FCPBX = nullptr;
	}

	if (FCDVR)
	{
		delete (caffe::shared_ptr<caffe::Net<float>>*)FCDVR;
		FCDVR = nullptr;
	}

	if (FCLDM)
	{
		delete (caffe::shared_ptr<caffe::Net<float>>*)FCLDM;
		FCLDM = nullptr;
	}

	if (bIsGoogleLoggingInitialized)
	{
		bIsGoogleLoggingInitialized = false;
		::google::ShutdownGoogleLogging();
	}
	return;
}

bool CFCFacedet::FCFInit(string strCaffeModelPath, bool bUseGPU)
{
	if (!bIsGoogleLoggingInitialized)
	{
		bIsGoogleLoggingInitialized = true;
		::google::InitGoogleLogging("FCFacedet");
	}

	if (bUseGPU)
	{
		char chMsg[100];
		sprintf(chMsg, "GPU Mode, Initializing..\n");
		printf(chMsg);
		Caffe::set_mode(Caffe::GPU);
	}
	else
	{
    char chMsg[100];
    sprintf(chMsg, "CPU Mode, Initializing..\n");
    printf(chMsg);
		Caffe::set_mode(Caffe::CPU);
	}

	caffe::shared_ptr<caffe::Net<float>>* PNet = new caffe::shared_ptr<caffe::Net<float>>;
	caffe::shared_ptr<caffe::Net<float>>* RNet = new caffe::shared_ptr<caffe::Net<float>>;
	caffe::shared_ptr<caffe::Net<float>>* ONet = new caffe::shared_ptr<caffe::Net<float>>;

	bool bRet = InitModels(strCaffeModelPath, *PNet, *RNet, *ONet);
	if (!bRet)
	{
		return false;
	}
	else
	{
		FCPBX = reinterpret_cast<void*>(PNet);
		FCDVR = reinterpret_cast<void*>(RNet);
		FCLDM = reinterpret_cast<void*>(ONet);
		return true;
	}
}

// Get a faces in image
bool CFCFacedet::FCFGetFaces(ST_IMAGE_DATA& stImgData, std::vector<ST_FACE_DATA>& vctFaces, int minFaceSize)
{
	int iType = CV_8UC3;
	if (stImgData.num_channels == 3)
	{
		iType = CV_8UC3;
	}
	else if (stImgData.num_channels == 1)
	{
		iType = CV_8UC1;
	}
	else
	{
		return false;
	}

	cv::Mat img(stImgData.height, stImgData.width, iType, stImgData.data);
	if (img.cols < 1 || img.rows < 1 || img.data == nullptr)
	{
		return false;
	}

	caffe::shared_ptr<caffe::Net<float>>* PNet = reinterpret_cast<caffe::shared_ptr<caffe::Net<float>>*>(FCPBX);
	caffe::shared_ptr<caffe::Net<float>>* RNet = reinterpret_cast<caffe::shared_ptr<caffe::Net<float>>*>(FCDVR);
	caffe::shared_ptr<caffe::Net<float>>* ONet = reinterpret_cast<caffe::shared_ptr<caffe::Net<float>>*>(FCLDM);

	return 	GetFaces(img, vctFaces, minFaceSize, *PNet, *RNet, *ONet);
}

// Release Memory allocated by this library
void CFCFacedet::FCFReleaseMemory(unsigned char* pData)
{
	if (pData)
	{
		delete[] pData;
		pData = nullptr;
	}
}

bool InitModels(std::string caffe_model_path,
	caffe::shared_ptr<caffe::Net<float>>& PNet,
	caffe::shared_ptr<caffe::Net<float>>& RNet,
	caffe::shared_ptr<caffe::Net<float>>& ONet)
{
	int PNet_num_channels_, RNet_num_channels_, ONet_num_channels_;
	cv::Size PNet_input_geometry_, RNet_input_geometry_, ONet_input_geometry_;

	// Load the network
	PNet.reset(new caffe::Net<float>(caffe_model_path + "/pnet.prototxt", TEST));
	PNet->CopyTrainedLayersFrom(caffe_model_path + "/pnet.caffemodel");
	if (PNet->num_inputs() != 1 || PNet->num_outputs() != 2)
	{
		return false;
	}

	caffe::Blob<float>* PNet_input_layer = PNet->input_blobs()[0];
	PNet_num_channels_ = PNet_input_layer->channels();
	PNet_input_geometry_ = cv::Size(PNet_input_layer->width(), PNet_input_layer->height());
	if (PNet_num_channels_ != 1 && PNet_num_channels_ != 3)
	{
		return false;
	}

	RNet.reset(new caffe::Net<float>(caffe_model_path + "/rnet.prototxt", TEST));
	RNet->CopyTrainedLayersFrom(caffe_model_path + "/rnet.caffemodel");
	if (RNet->num_inputs() != 1 || RNet->num_outputs() != 2)
	{
		return false;
	}

	caffe::Blob<float>* RNet_input_layer = RNet->input_blobs()[0];
	RNet_num_channels_ = RNet_input_layer->channels();
	RNet_input_geometry_ = cv::Size(RNet_input_layer->width(), RNet_input_layer->height());
	if (RNet_num_channels_ != 1 && RNet_num_channels_ != 3)
	{
		return false;
	}

	caffe::Blob<float>* ONet_input_layer = nullptr;

	ONet.reset(new caffe::Net<float>(caffe_model_path + "/onet.prototxt", TEST));
	ONet->CopyTrainedLayersFrom(caffe_model_path + "/onet.caffemodel");

	if (ONet->num_inputs() != 1 || ONet->num_outputs() != 3)
	{
		return false;
	}

	ONet_input_layer = ONet->input_blobs()[0];
	ONet_num_channels_ = ONet_input_layer->channels();
	ONet_input_geometry_ = cv::Size(ONet_input_layer->width(), ONet_input_layer->height());
	if (ONet_num_channels_ != 1 && ONet_num_channels_ != 3)
	{
		return false;
	}

	return true;
}

bool GetFaces(cv::Mat& img, std::vector<ST_FACE_DATA>& vctFaces, int minFaceSize,
	caffe::shared_ptr<caffe::Net<float>>& PNet,
	caffe::shared_ptr<caffe::Net<float>>& RNet,
	caffe::shared_ptr<caffe::Net<float>>& ONet)
{
	//steps's threshold
	float threshold[3] = { 0.650, 0.700, 0.700 };
	//scale factor
	float factor = 0.7;

	vector<RESULT> result;

#ifdef DEBUG_PRINT_TIME
	clock_t start1, end1;
	start1 = clock();
#endif

	result = detect_face(img, minFaceSize, PNet, RNet, ONet, threshold, false, factor);

	int ifacecount = result.size();

#ifdef DEBUG_PRINT_TIME
	end1 = (double)(1000 * (clock() - start1) / CLOCKS_PER_SEC);
	cout << "FaceCnt: " << ifacecount << ",  Time_Det:  " << end1 << "ms" << endl;
#endif

	if (ifacecount < 1)
	{
		//img = img.t();
		//imshow("result", img);
		//cvWaitKey(10);
		return false;
	}

	// get faces data
	for (unsigned int u = 0; u < result.size(); u++)
	{
		ST_FACE_DATA stFaceData;
		stFaceData.x_pos = result[u].boudingboxes.y;
		stFaceData.y_pos = result[u].boudingboxes.x;
		stFaceData.width = result[u].boudingboxes.height;
		stFaceData.height = result[u].boudingboxes.width;
		stFaceData.num_channels = img.channels();

		for (int i = 0; i < 5; ++i) {
			stFaceData.landmark[i * 2] = result[u].points[i].y;
			stFaceData.landmark[i * 2 + 1] = result[u].points[i].x;
		}

		vctFaces.push_back(stFaceData);
	}

	return true;
}
