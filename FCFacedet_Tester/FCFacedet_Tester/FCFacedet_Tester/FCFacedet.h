#ifndef __FCFACEDET_H__
#define __FCFACEDET_H__

#include <vector>
#include <string>
#include <string.h>

#define FCFACEDET_EXPORTS

#ifdef WIN32
#ifdef FCFACEDET_EXPORTS
#define FCFACEDET_API __declspec(dllexport)
#else
#define FCFACEDET_API __declspec(dllimport)
#endif
#else
#define FCFACEDET_API
#endif

using namespace std;

typedef struct _st_image_data
{
	_st_image_data()
	{
		data = nullptr;
		width = 0;
		height = 0;
		num_channels = 0;
	}

	_st_image_data(int img_width,
				   int img_height,
				   int img_num_channels = 1)
	{
		data = nullptr;
		width = img_width;
		height = img_height;
		num_channels = img_num_channels;
	}

	unsigned char* data;
	int width;
	int height;
	int num_channels;
} ST_IMAGE_DATA;


typedef struct _st_face_data {
	_st_face_data() {
		data = nullptr;
		data_size = 0;
		x_pos = 0;
		y_pos = 0;
		width = 0;
		height = 0;
		num_channels = 0;
		memset(landmark, 0, sizeof(landmark));
	}

	_st_face_data(int face_x_pos, int face_y_pos, int face_width, int face_height,
		int img_num_channels = 1) {
		data = nullptr;
		data_size = 0;
		x_pos = face_x_pos;
		y_pos = face_y_pos;
		width = face_width;
		height = face_height;
		num_channels = img_num_channels;
	}

	unsigned char* data;
	int data_size;
	float landmark[10];
	int x_pos;
	int y_pos;
	int width;
	int height;
	int num_channels;
} ST_FACE_DATA;

#ifdef __cplusplus
extern "C" {
#endif
// This class is exported from the FCFacerec.dll
class FCFACEDET_API CFCFacedet {
public:
	CFCFacedet(void);
	~CFCFacedet();

	// Library Initialize
	bool FCFInit(std::string strCaffeModelPath = "./model", bool bUseGPU = true);

	// Get faces in image
	bool FCFGetFaces(ST_IMAGE_DATA& stImgData, std::vector<ST_FACE_DATA>& vctFaces, int minFaceSize = 40);

	// Release Memory allocated by this library
	void FCFReleaseMemory(unsigned char* pData);

private:
	void* FCPBX;
	void* FCDVR;
	void* FCLDM;
};

#ifdef __cplusplus
}
#endif

#endif
